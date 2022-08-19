import numpy as np 
import torch
from matrix_groups.triangular import MUp
from optimizers.noisy_optimizer import *


class RankCov(NoisyOptimizer):
    def __init__(self, params, data_size, rank=1, lr=1e-3, momentum_grad=0.9, damping=0.01, mc_samples=1,
                 prior_precision=0.4, gamma=1, device='cuda', momentum_hess=None, hess_init=None):
        """

        :param params:
        :param data_size:
        :param k:
        :param lr:
        :param mc_samples:
        :param eta:
        :param damping:
        :param beta:
        :param gamma:
        :param device:
        """
        assert data_size >= 1
        assert (type(rank) == int) and (rank >= 0)
        assert lr > 0.0
        assert momentum_grad >= 0.0
        assert mc_samples >= 1
        assert prior_precision > 0.0
        assert damping >= 0.0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr

        self.data_size = data_size
        self.mc_samples = mc_samples
        self.gamma = gamma
        self.device = device
        
        defaults = dict(lr=lr, data_size=data_size, rank=rank, momentum_grad=momentum_grad, mc_samples=mc_samples,
                        momentum_hess=momentum_hess, damping=damping, prior_precision=prior_precision)
        super(NoisyOptimizer, self).__init__(params, defaults)

        self._init_momentum_buffers()
        self._reset_param_and_grad_samples(hess_init)

        self.d = np.sum(self.d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {self.d_is}")

    def _init_momentum_buffers(self, hess_init):
        self.d_is = []
        for group in self.param_groups:
            params = group['params']
            eta = group['prior_precision']
            k = group['rank']
            n = group['data_size']
            damping = group['damping']
            mc_samples = group['mc_samples']
            for i, p in enumerate(params):
                state = self.state[p]
                d_i = int(np.prod(p.shape))
                self.d_is.append(d_i)
                state['momentum'] = torch.zeros(d_i, 1, device=self.device)

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                identity = MUp.eye(d_i,
                                   np.minimum(d_i, k),
                                   device=self.device)
                if hess_init is None:
                    state['scale'] = np.sqrt(eta / n + damping) * identity
                else:
                    state['scale'] = hess_init * identity
                # For gradient sample collection
                state['grad_samples'] = torch.zeros((mc_samples, np.prod(p.shape), 1), device=self.device)
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

    @torch.no_grad()
    def sample(self):
        """
        Return list z in shape [num_layers, M, *param_dim]
        """
        for group in self.param_groups:
            n = group['data_size']
            damping = group['damping']
            mc_samples = group['mc_samples']
            for p in group['params']:
                state = self.state[p]
                # z_i ~ N(0, n (B B^T)^{-1})
                state['param_samples'] = (torch.sqrt(torch.tensor(n)) * state['scale'].add_id(np.sqrt(damping))).sample(mu=p.reshape((-1, 1)), n=mc_samples).reshape((mc_samples, -1, 1))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        self.sample() # (2)
        self._stash_param_averages()

        # Save params for each param group
        # Sample for each param group and parameter, and store the sample
        # Compute closure and save gradients
        # Restore original parameters and update parameters
        if closure is None:
            raise ValueError("Closure needs to be specified for Noisy Optimizer!")
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
                    state['grad_samples'][i] = p.grad.reshape((-1, 1))
        avg_pred = torch.mean(torch.stack(outputs, dim=0), axis=0)
        avg_loss = torch.mean(torch.stack(losses, dim=0))

        # Restore model parameters to original state
        self._restore_params()

        # Update parameters
        self._update()
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

                mu = p.reshape((-1, 1))
                g_bar = torch.mean(state['grad_samples'], axis=0)
                g_mu = gamma * eta / n * mu + g_bar # (3)
                state['momentum'].mul_(beta1).add_((1 - beta1) * g_mu) # (4)

                t = state['step']
                m_bar = state['momentum'] / (1 - beta1 ** (t + 1)) # (5)
                # B_bar = torch.rsqrt(torch.tensor(1 - beta2 ** t + eps)) * state['scale'].add_id(-np.sqrt(eta / n + damping))
                B_bar = state['scale'].add_id(np.sqrt(eta / n + damping))

                # Evaluate matrix vector products so intermediate results stay in memory
                update = (B_bar.inv().t() @ (B_bar.inv() @ m_bar)).reshape(p.shape) # (7)
                p.add_(-lr * update)

                state['scale'] = state['scale'].add_id(np.sqrt(damping)).update(beta2, eta, n, state['grad_samples'], state['param_samples'] - mu) # (6) & (8)

                # update the steps for each param group update
                state['step'] += 1

    def _stash_param_averages(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_average'] = p.data

    def _restore_params(self):
        for group in self.param_groups:
            for p in group['params']:
                p.data = self.state[p]['param_average']
    
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
                    for p, B in zip(group['params'], group['scales']):
                        kl_divergence += self.eta * (B.inv().frobenius_norm() + torch.sum(p ** 2)) - 2 * B.log_det()
                kl_divergence *= 0.5
            else:
                kl_divergence = 0.0
            # Sample predictions of shape (M, BATCH_SIZE, *) and reshape into 
            # (M * BATCH_SIZE, *) to compute average of sum of losses 
            # 1/M sum_i^M (sum_j^N l_ij) = N * 1/(M * N) sum_i^{M * N} l_i
            return self.N * loss_fn(preds, labels) - gamma * kl_divergence
        return f
