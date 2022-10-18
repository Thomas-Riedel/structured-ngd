import numpy as np 
import torch
from torch import Tensor
from torch.nn import Parameter
from matrix_groups.triangular import MUp, MLow
from typing import Union, Tuple, Callable
from optimizers.noisy_optimizer import *
import warnings


class StructuredNGD(NoisyOptimizer):
    def __init__(self, params, data_size: int, k: int = 0, lr: float = 1e-1, mc_samples: int = 1,
                 momentum_grad: float = None, momentum_prec: Union[float] = None, damping: float = None,
                 prior_precision: float = None, gamma: float = None, prec_init: Union[float] = None,
                 structure: str = 'rank_cov', debias: bool = False) -> None:
        """Structured NGD Optimizer inheriting from torch.optim.Optimizer

        :param params: torch.nn.Parameters of a model to be optimized using NGD
        :param data_size: int, training set size
        :param k: int, rank for structure 'rank_cov' or arrow width for structure 'arrowhead', resp.
        :param lr: float, learning rate
        :param momentum_grad: float, first order momentum strength
        :param momentum_prec: second order momentum strength
        :param damping: float, damping strength for inversion or linear system solutions
        :param mc_samples: int, number of Monte Carlo samples
        :param prior_precision: float, prior precision
        :param gamma: float, regularization strength for ELBO
        :param device: str, torch device to run operations on (GPU or CPU)
        :param prec_init: np.array, initial value for square root precision
        :param structure: str, covariance structure
        :param debias: bool, whether to debias square root precision updates
        """
        assert data_size >= 1
        assert (type(k) == int) and (k >= 0)
        assert lr > 0.0
        assert mc_samples >= 1

        if structure not in ['rank_cov', 'arrowhead']:
            raise NotImplementedError()
        if momentum_grad is None:
            momentum_grad = 0.6
        if momentum_prec is None:
            momentum_prec = 0.999
        if prior_precision is None:
            prior_precision = 0.4
        if damping is None:
            damping = 0.01
        if gamma is None:
            gamma = 1.0

        assert momentum_grad >= 0.0
        assert momentum_prec >= 0.0
        assert prior_precision > 0.0
        assert damping >= 0.0

        self.structure = structure
        self.data_size = data_size
        self.mc_samples = mc_samples
        self.gamma = gamma
        self.debias = debias
        self.__name__ = f"StructuredNGD (structure = {structure}, k = {k}"
        if self.mc_samples > 1:
            self.__name__ += f", M = {mc_samples}"
        self.__name__ += ')'

        defaults = dict(lr=lr, data_size=data_size, k=k, momentum_grad=momentum_grad, mc_samples=mc_samples,
                        momentum_prec=momentum_prec, damping=damping, prior_precision=prior_precision)
        super(NoisyOptimizer, self).__init__(params, defaults)

        self._init_momentum_buffers(prec_init)
        self._reset_param_and_grad_samples()

        self.d = np.sum(self.d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {self.d_is}")
        print(f"k = {k}; structure = {structure}")

    def _init_momentum_buffers(self, prec_init: float) -> None:
        """Initialize square root precision with initial value if specified, otherwise as prior precision

        :param prec_init: np.array, optional initial value for square root precision
        """
        self.d_is = []
        for group in self.param_groups:
            params = group['params']
            eta = group['prior_precision']
            k = group['k']
            n = group['data_size']
            damping = group['damping']
            for i, p in enumerate(params):
                state = self.state[p]
                d_i = p.nelement()
                self.d_is.append(d_i)
                state['momentum_grad_buffer'] = torch.zeros(d_i, device=p.device)

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                if self.structure == 'rank_cov':
                    identity = MUp.eye(d_i,
                                       np.minimum(d_i, k),
                                       device=p.device,
                                       damping=damping)
                else:
                    identity = MLow.eye(d_i,
                                        np.minimum(d_i, k),
                                        device=p.device,
                                        damping=damping)
                if prec_init is None:
                    state['momentum_prec_buffer'] = torch.sqrt(torch.tensor(eta / n + self.debias * damping)) * identity
                else:
                    state['momentum_prec_buffer'] = prec_init * identity
                # t counter
                state['step'] = 0
                if k > d_i:
                    warnings.warn(f"The rank parameter {k} at layer {i} is bigger than the number of parameters {d_i} "
                                  f"and will be capped, i.e. k_{i} := {d_i}.")

    def _reset_param_and_grad_samples(self) -> None:
        """Reset parameter and gradient samples.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_samples'] = torch.zeros((self.mc_samples, p.nelement()), device=p.device)
                    self.state[p]['grad_samples'] = torch.zeros((self.mc_samples, p.nelement()), device=p.device)

    @torch.no_grad()
    def step(self, closure: Callable = None) -> Tuple[float, Tensor]:
        """Performs a single optimization step on the parameters.

        :param closure: Callable, perform forward step for model, backpropagate and return loss
        :return:
            Tuple(avg_loss, avg_pred)
                avg_loss: float, average loss over all MC samples
                avg_pred: torch.Tensor: average prediction over all MC samples
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

        for i in range(self.mc_samples):
            self._sample_weight_and_collect(i)  # (2)
            with torch.enable_grad():
                loss, output = closure()
            losses.append(loss.detach())
            outputs.append(output.detach())
            self._collect_grad_samples(i)

        # Update parameters
        self._update()

        # Restore model parameters to original state
        self._restore_param_averages()

        # Clear sample buffers
        self._reset_param_and_grad_samples()

        # Average losses and predictions over all MC samples and return
        avg_loss = torch.mean(torch.stack(losses, dim=0))
        avg_pred = torch.mean(torch.stack(outputs, dim=0), axis=0)
        return avg_loss, avg_pred

    def _stash_param_averages(self) -> None:
        """Stash parameters in field 'param_average'.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['param_average'] = p.data

    def _sample_weight_and_collect(self, sample_index: int) -> None:
        """Sample parameters from posterior distribution according to
            z_i = p_avg + 1/sqrt(n) * B^{-T} eps ~ N(p_avg, n * (B B^T)^{-1}),
        where eps ~ N(0, I) and save in 'param_samples'.
        """
        for group in self.param_groups:
            n = group['data_size']
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    b_bar = state['momentum_prec_buffer']
                    p_avg = state['param_average'].reshape(-1)
                    # z_i ~ N(mu, n (B B^T)^{-1})
                    # z_i = mu + 1/sqrt(n) * B^{-T} eps, with eps ~ N(0, I)
                    d = p.nelement()
                    eps = torch.randn(d, device=p.device)
                    p_sample = p_avg + torch.rsqrt(torch.tensor(n)) * b_bar.transpose_solve(eps)
                    p.data = p_sample.reshape(p.shape)
                    state['param_samples'][sample_index] = p_sample.reshape(-1)

    def _sample_weight(self) -> None:
        """Sample parameters from posterior distribution according to
            z_i = p_avg + 1/sqrt(n) * B^{-T} eps ~ N(p_avg, n * (B B^T)^{-1}),
        where eps ~ N(0, I).
        """
        for group in self.param_groups:
            n = group['data_size']
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    b_bar = state['momentum_prec_buffer']
                    p_avg = state['param_average'].reshape(-1)
                    # z_i ~ N(mu, n (B B^T)^{-1})
                    # z_i = mu + 1/sqrt(n) * B^{-T} eps, with eps ~ N(0, I)
                    d = p.nelement()
                    eps = torch.randn(d, device=p.device)
                    p_sample = p_avg + torch.rsqrt(torch.tensor(n)) * b_bar.transpose_solve(eps)
                    p.data = p_sample.reshape(p.shape)

    def _collect_grad_samples(self, sample_index: int) -> None:
        """Collect a single gradient sample for all parameters and save it in 'grad_samples'
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['grad_samples'][sample_index] = p.grad.reshape(-1)

    def _update(self) -> None:
        """Update momentum, parameter averages and square root precisions
        """
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['momentum_grad'], group['momentum_prec']
            n = group['data_size']
            eta = group['prior_precision']
            damping = group['damping']
            # For each layer
            for i, p in enumerate(group['params']):
                if p.requires_grad:
                    self._update_momentum_grad_buffers(p, eta, n, beta1)
                    self._update_param_averages(p, lr, beta1, beta2, eta, n, damping)
                    self._update_momentum_prec_buffers(p, eta, n, beta2)

    def _update_momentum_grad_buffers(self, p: Parameter, eta: float, n: int, beta1: float) -> None:
        """Update momentum term:
            momentum <- beta * momentum + (1 - beta) * g_mu

        :param p: torch.nn.Parameter used to index momentum for update
        :param eta: float, prior precision
        :param n: int, training set size
        :param beta1: float, first order momentum strength
        """
        state = self.state[p]
        p_avg = state['param_average'].reshape(-1)
        g_bar = torch.mean(state['grad_samples'], axis=0)
        g_mu = self.gamma * eta / n * p_avg + g_bar  # (3)
        state['momentum_grad_buffer'].mul_(beta1).add_(g_mu, alpha=1-beta1)  # (4)

    def _update_param_averages(self, p: Parameter, lr: float, beta1: float, beta2: float,
                               eta: float, n: int, damping: float) -> None:
        """Update parameter average for parameter p
            p <- p - lr * B^{-T} * B^{-1} * momentum

        :param p: torch.nn.Parameter used to index parameter average to be updated
        :param lr: float, learning rate
        :param beta1: float, first order momentum strength
        :param beta2: float, second order momentum strength
        :param eta: float, prior precision
        :param n: int, training set size
        :param damping: float, damping strength for matrix inversion or linear system solution
        """
        state = self.state[p]
        m_bar = state['momentum_grad_buffer']
        b_bar = state['momentum_prec_buffer']
        if self.debias:
            eps = 1e-9
            t = self.state[p]['step']
            m_bar = m_bar / (1 - beta1 ** (t + 1))  # (5)
            b_bar = torch.rsqrt(torch.tensor(1 - beta2 ** t + eps)) * b_bar.add_id(-torch.sqrt(eta / n + damping))
            b_bar = b_bar.add_id(torch.sqrt(eta / n + damping))
            state['step'] += 1

        # Evaluate matrix vector products so intermediate results stay in memory
        update = b_bar.transpose_solve(b_bar.solve(m_bar)).reshape(p.shape) # (7)
        state['param_average'].add_(-lr * update)

    def _update_momentum_prec_buffers(self, p: Parameter, eta: float, n: int, beta2: float) -> None:
        """Update square root precisions for parameter p
            B <- B @ h((1-beta) * C .* kappa(B^{-1} G_S B^{-T}))

        :param p: torch.nn.Parameter used to index square root precision associated with p
        :param eta: float, prior precision
        :param n: int, training set size
        :param beta2: float, second order momentum strength
        """
        state = self.state[p]
        p_avg = state['param_average'].reshape((-1, 1))
        b_bar = state['momentum_prec_buffer']
        grad_samples = state['grad_samples'].unsqueeze(-1)
        param_samples = state['param_samples'].unsqueeze(-1)
        state['momentum_prec_buffer'] = b_bar._update(beta2, eta, n, grad_samples,
                                                      param_samples - p_avg)  # (6) & (8)

    def _restore_param_averages(self) -> None:
        """Save 'param_average' field back to parameters
        """
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
                    for p, B in zip(group['params'], group['momentum_prec_buffer']):
                        kl_divergence += self.eta * (B.inv().frobenius_norm() + torch.sum(p ** 2)) - 2 * B.log_det()
                kl_divergence *= 0.5
            else:
                kl_divergence = 0.0
            # Sample predictions of shape (M, BATCH_SIZE, *) and reshape into 
            # (M * BATCH_SIZE, *) to compute average of sum of losses 
            # 1/M sum_i^M (sum_j^N l_ij) = N * 1/(M * N) sum_i^{M * N} l_i
            return self.N * loss_fn(preds, labels) - gamma * kl_divergence
        return f
