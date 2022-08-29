import numpy as np 
import torch
from matrix_groups.triangular import MUp, MLow
from optimizers.noisy_optimizer import *


class StructuredNGD(NoisyOptimizer):
    def __init__(self, params, data_size, k=0, lr=1e-3, momentum_grad=0.9, damping=0.01, mc_samples=1,
                 prior_precision=0.4, gamma=1, device='cuda', momentum_hess=None, hess_init=None, method='arrowhead', debias=False):
        assert data_size >= 1
        assert (type(k) == int) and (k >= 0)
        assert lr > 0.0
        assert momentum_grad >= 0.0
        assert mc_samples >= 1
        assert prior_precision > 0.0
        assert damping >= 0.0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr

        if not method in ['rank_cov', 'arrowhead']:
            raise NotImplementedError()

        self.method = method
        self.data_size = data_size
        self.mc_samples = mc_samples
        self.gamma = gamma
        self.device = device
        self.debias = debias
        
        defaults = dict(lr=lr, data_size=data_size, k=k, momentum_grad=momentum_grad, mc_samples=mc_samples,
                        momentum_hess=momentum_hess, damping=damping, prior_precision=prior_precision)
        super(NoisyOptimizer, self).__init__(params, defaults)

        self._init_momentum_buffers(method, hess_init)
        self._reset_param_and_grad_samples()

        self.d = np.sum(self.d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {self.d_is}")
        print(f"k = {k}; method = {method}")

    def _init_momentum_buffers(self, method, hess_init):
        self.d_is = []
        for group in self.param_groups:
            params = group['params']
            eta = group['prior_precision']
            k = group['k']
            n = group['data_size']
            damping = group['damping']
            mc_samples = group['mc_samples']
            for i, p in enumerate(params):
                state = self.state[p]
                d_i = int(np.prod(p.shape))
                self.d_is.append(d_i)
                state['momentum_grad_buffer'] = torch.zeros(d_i, 1, device=self.device)

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                if method == 'rank_cov':
                    identity = MUp.eye(d_i,
                                       np.minimum(d_i, k),
                                       device=self.device)
                else:
                    identity = MLow.eye(d_i,
                                       np.minimum(d_i, k),
                                       device=self.device)
                if hess_init is None:
                    state['momentum_hess_buffer'] = np.sqrt(eta / n + damping) * identity
                else:
                    state['momentum_hess_buffer'] = hess_init * identity
                # t counter
                state['step'] = 0
                if k > d_i:
                    raise Warning(f"The rank parameter {k} at layer {i} is bigger than the number of parameters {d_i} "
                                  f"and will be capped, i.e. k_{i} := {d_i}.")

    def _reset_param_and_grad_samples(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_samples'] = []
                    self.state[p]['grad_samples'] = []

    def _sample_params(self):
        """
        Return list z in shape [num_layers, M, *param_dim]
        """
        for group in self.param_groups:
            n = group['data_size']
            damping = group['damping']
            mc_samples = group['mc_samples']
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # z_i ~ N(0, n (B B^T)^{-1})
                    state['param_samples'] = (torch.sqrt(torch.tensor(n)) * state['momentum_hess_buffer'].add_id(np.sqrt(damping))).sample(mu=p.reshape((-1, 1)), n=mc_samples).reshape((mc_samples, -1, 1))

    def _sample_grads(self, closure):
        outputs = []
        losses = []
        for i in range(self.mc_samples):
            for group in self.param_groups:
                for p in group['params']:
                    z = self.state[p]['param_samples']
                    z_i = z[i].reshape(p.shape)
                    p.data = z_i
            with torch.enable_grad():
                loss, output = closure()

            losses.append(loss.detach())
            outputs.append(output.detach())
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['grad_samples'].append(p.grad.reshape((-1, 1)))
            return losses, outputs

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        self._stash_param_averages()
        self._sample_params() # (2)

        # Save params for each param group
        # Sample for each param group and parameter, and store the sample
        # Compute closure and save gradients
        # Restore original parameters and update parameters
        if closure is None:
            raise ValueError("Closure needs to be specified for Noisy Optimizer!")
        losses, outputs = self._sample_grads(closure)
        # Update parameters
        self._update()

        # Restore model parameters to original state
        self._restore_params()

        # Clear sample buffers
        self._reset_param_and_grad_samples()

        avg_pred = torch.mean(torch.stack(outputs, dim=0), axis=0)
        avg_loss = torch.mean(torch.stack(losses, dim=0))
        return avg_loss, avg_pred

    def _update(self):
        gamma = self.gamma
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['momentum_grad'], group['momentum_hess']
            n = group['data_size']
            eta = group['prior_precision']
            damping = group['damping']

            # For each layer
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                state['grad_samples'] = torch.stack(state['grad_samples'], dim=0)

                mu = p.reshape((-1, 1))
                g_bar = torch.mean(state['grad_samples'], axis=0)
                g_mu = gamma * eta / n * mu + g_bar # (3)
                state['momentum_grad_buffer'].mul_(beta1).add_((1 - beta1) * g_mu) # (4)

                m_bar = state['momentum_grad_buffer']
                b_bar = state['momentum_hess_buffer']
                if self.debias:
                    eps = 1e-9
                    t = state['step']
                    m_bar = m_bar / (1 - beta1 ** (t + 1)) # (5)
                    b_bar = torch.rsqrt(torch.tensor(1 - beta2 ** t + eps)) * b_bar.add_id(-np.sqrt(eta / n + damping))
                    b_bar = b_bar.add_id(np.sqrt(eta / n + damping))
                    state['step'] += 1

                b_bar = b_bar.add_id(np.sqrt(damping))

                # Evaluate matrix vector products so intermediate results stay in memory
                update = (b_bar.inv().t() @ (b_bar.inv() @ m_bar)).reshape(p.shape) # (7)
                p.add_(-lr * update)

                state['momentum_hess_buffer'] = state['momentum_hess_buffer'].add_id(np.sqrt(damping))._update(beta2, eta, n, state['grad_samples'], state['param_samples'] - mu) # (6) & (8)

    def _stash_param_averages(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_average'] = p.data

    def _restore_params(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.data = self.state[p]['param_average']
                    self.state[p]['param_average'] = None
    
    def elbo(self, loss_fn):
        def f(preds, labels, gamma=1):
            """Compute ELBO: L(mu, Sigma) = E_q[sum l_i] - gamma * KL(q || p)
            """
            # When q, p both MVG, we have
            # (see https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions )
            # KL(q || p) = 1/2(eta * ||B^{-1}||_F^2 - d + eta * ||mu||^2 - d ln(eta) - 2 ln(det(B))
            # Otherwise, this value needs to be approximated through sampling
            # For the expected value of the summed losses, take M samples and approximate by averaging
            if gamma > 0:
                kl_divergence = -self.d * (1 + torch.log(torch.tensor(self.eta) + torch.finfo().tiny))
                for group in self.param_groups:
                    for p, B in zip(group['params'], group['momentum_hess_buffer']):
                        kl_divergence += self.eta * (B.inv().frobenius_norm() + torch.sum(p ** 2)) - 2 * B.log_det()
                kl_divergence *= 0.5
            else:
                kl_divergence = 0.0
            # Sample predictions of shape (M, BATCH_SIZE, *) and reshape into 
            # (M * BATCH_SIZE, *) to compute average of sum of losses 
            # 1/M sum_i^M (sum_j^N l_ij) = N * 1/(M * N) sum_i^{M * N} l_i
            return self.N * loss_fn(preds, labels) - gamma * kl_divergence
        return f
