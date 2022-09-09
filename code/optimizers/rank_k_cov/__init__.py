import numpy as np 
import torch
from torch import Tensor
from torch.nn import Parameter
from matrix_groups.triangular import MUp, MLow
from typing import Union, Tuple, Callable
from optimizers.noisy_optimizer import *


class StructuredNGD(NoisyOptimizer):
    def __init__(self, params, data_size: int, k: int = 0, lr: float = 1e-3, momentum_grad: float = 0.9,
                 damping: float = 0.01, mc_samples: int = 1, prior_precision: float = 100, gamma: float = 1,
                 device: str = 'cuda', momentum_hess: Union[float] = None,
                 hess_init: Union[float] = None, structure: str = 'rank_cov', debias: bool = False) -> None:
        assert data_size >= 1
        assert (type(k) == int) and (k >= 0)
        assert lr > 0.0
        assert momentum_grad >= 0.0
        assert mc_samples >= 1
        assert prior_precision > 0.0
        assert damping >= 0.0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr

        if structure not in ['rank_cov', 'arrowhead']:
            raise NotImplementedError()

        self.structure = structure
        self.data_size = data_size
        self.mc_samples = mc_samples
        self.gamma = gamma
        self.device = device
        self.debias = debias
        
        defaults = dict(lr=lr, data_size=data_size, k=k, momentum_grad=momentum_grad, mc_samples=mc_samples,
                        momentum_hess=momentum_hess, damping=damping, prior_precision=prior_precision)
        super(NoisyOptimizer, self).__init__(params, defaults)

        self._init_momentum_buffers(hess_init)
        self._reset_param_and_grad_samples()

        self.d = np.sum(self.d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {self.d_is}")
        print(f"k = {k}; structure = {structure}")

    def _init_momentum_buffers(self, hess_init: float) -> None:
        self.d_is = []
        for group in self.param_groups:
            params = group['params']
            eta = group['prior_precision']
            k = group['k']
            n = group['data_size']
            damping = group['damping']
            for i, p in enumerate(params):
                state = self.state[p]
                d_i = int(np.prod(p.shape))
                self.d_is.append(d_i)
                state['momentum_grad_buffer'] = torch.zeros(d_i, 1, device=self.device)

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                if self.structure == 'rank_cov':
                    identity = MUp.eye(d_i,
                                       np.minimum(d_i, k),
                                       device=self.device,
                                       damping=np.sqrt(damping))
                else:
                    identity = MLow.eye(d_i,
                                        np.minimum(d_i, k),
                                        device=self.device,
                                        damping=np.sqrt(damping))
                if hess_init is None:
                    state['momentum_hess_buffer'] = np.sqrt(eta / n + damping) * identity
                else:
                    state['momentum_hess_buffer'] = hess_init * identity
                # t counter
                state['step'] = 0
                if k > d_i:
                    raise Warning(f"The rank parameter {k} at layer {i} is bigger than the number of parameters {d_i} "
                                  f"and will be capped, i.e. k_{i} := {d_i}.")

    def _reset_param_and_grad_samples(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_samples'] = []
                    self.state[p]['grad_samples'] = []

    @torch.no_grad()
    def step(self, closure: Callable = None) -> Tuple[float, Tensor]:
        """Performs a single optimization step.
        """
        self._stash_param_averages()

        # Save params for each param group
        # Sample for each param group and parameter, and store the sample
        # Compute closure and save gradients
        # Restore original parameters and update parameters
        if closure is None:
            raise ValueError("Closure needs to be specified for Noisy Optimizer!")

        losses = []
        outputs = []
        for _ in range(self.mc_samples):
            self._sample_weight_and_collect()  # (2)
            with torch.enable_grad():
                loss, output = closure()
            losses.append(loss.detach())
            outputs.append(output.detach())
            self._collect_grad_samples()
        # Update parameters
        self._update()

        # Restore model parameters to original state
        self._restore_param_averages()

        # Clear sample buffers
        self._reset_param_and_grad_samples()

        avg_pred = torch.mean(torch.stack(outputs, dim=0), axis=0)
        avg_loss = torch.mean(torch.stack(losses, dim=0))
        return avg_loss, avg_pred

    def _stash_param_averages(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_average'] = p.data

    def _sample_weight_and_collect(self) -> None:
        for group in self.param_groups:
            n = group['data_size']
            for p in group['params']:
                if p.requires_grad:
                    b_bar = self.state[p]['momentum_hess_buffer']
                    p_avg = self.state[p]['param_average'].reshape((-1, 1))
                    # z_i ~ N(mu, n (B B^T)^{-1})
                    # z_i = mu + 1/sqrt(n) * B^{-T} eps, with eps ~ N(0, I)
                    d = int(np.prod(p.shape))
                    eps = torch.randn((d, 1), device=self.device)
                    p_sample = p_avg + torch.rsqrt(torch.tensor(n)) * b_bar.t().solve(eps)
                    p.data = p_sample.reshape(p.shape)
                    self.state[p]['param_samples'].append(p_sample.reshape((-1, 1)))

    def _collect_grad_samples(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['grad_samples'].append(p.grad.reshape((-1, 1)))

    def _stack_samples(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_samples'] = torch.stack(self.state[p]['param_samples'], dim=0)
                    self.state[p]['grad_samples'] = torch.stack(self.state[p]['grad_samples'], dim=0)

    def _update(self) -> None:
        self._stack_samples()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['momentum_grad'], group['momentum_hess']
            n = group['data_size']
            eta = group['prior_precision']
            damping = group['damping']
            # For each layer
            for i, p in enumerate(group['params']):
                if p.requires_grad:
                    self._update_momentum_grad_buffers(p, eta, n, beta1)
                    self._update_param_averages(p, lr, beta1, beta2, eta, n, damping)
                    self._update_momentum_hess_buffers(p, eta, n, damping, beta2)

    def _update_momentum_grad_buffers(self, p: Parameter, eta: float, n: int, beta1: float) -> None:
        p_avg = self.state[p]['param_average'].reshape((-1, 1))
        g_bar = torch.mean(self.state[p]['grad_samples'], axis=0)
        g_mu = self.gamma * eta / n * p_avg + g_bar  # (3)
        self.state[p]['momentum_grad_buffer'].mul_(beta1).add_((1 - beta1) * g_mu)  # (4)

    def _update_param_averages(self, p: Parameter, lr: float, beta1: float, beta2: float,
                               eta: float, n: int, damping: float) -> None:
        m_bar = self.state[p]['momentum_grad_buffer']
        b_bar = self.state[p]['momentum_hess_buffer']
        if self.debias:
            eps = 1e-9
            t = self.state[p]['step']
            m_bar = m_bar / (1 - beta1 ** (t + 1))  # (5)
            b_bar = torch.rsqrt(torch.tensor(1 - beta2 ** t + eps)) * b_bar.add_id(-np.sqrt(eta / n + damping))
            b_bar = b_bar.add_id(np.sqrt(eta / n + damping))
            self.state[p]['step'] += 1

        # Evaluate matrix vector products so intermediate results stay in memory
        update = b_bar.t().solve(b_bar.solve(m_bar)).reshape(p.shape) # (7)
        self.state[p]['param_average'].add_(-lr * update)

    def _update_momentum_hess_buffers(self, p: Parameter, eta: float, n: int, damping: float, beta2: float) -> None:
        p_avg = self.state[p]['param_average'].reshape((-1, 1))
        b_bar = self.state[p]['momentum_hess_buffer']
        grad_samples = self.state[p]['grad_samples']
        param_samples = self.state[p]['param_samples']
        self.state[p]['momentum_hess_buffer'] = b_bar._update(beta2, eta, n, grad_samples, param_samples - p_avg)  # (6) & (8)

    def _restore_param_averages(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.data = self.state[p]['param_average']
                    self.state[p]['param_average'] = None

    def elbo(self, loss_fn: Callable) -> Callable:
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
