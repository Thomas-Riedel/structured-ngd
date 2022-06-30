import numpy as np 
import torch
from torch.optim import Optimizer
from matrix_groups.triangular import B_up


class Rank_kCov(Optimizer):
    def __init__(self, params, N, k=1, lr=1e-3, eta=5, betas=(0.9, 0.999),
                 weight_decay=0, device='cuda'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.N = N
        self.eta = eta
        self.device = device
        
        defaults = dict(lr=lr, betas=betas,
                        weight_decay=weight_decay)
        super(Rank_kCov, self).__init__(params, defaults)

        d_is = []
        for group in self.param_groups:
            params = group['params']
            group['momentum'], group['scales'] = [], []
            for p in params:
                d_i = np.prod(p.shape)
                d_is.append(d_i)
                group['momentum'].append(torch.zeros(d_i, 1).to(self.device))

                # Initialize precision S to prior precision N / eta * I, 
                # i.e., B = sqrt(N / eta) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                B_init = np.sqrt(N / eta) * B_up.eye(d_i, 
                                                     np.minimum(d_i, k),
                                                     device=self.device)
                group['scales'].append(B_init)
        self.d = np.sum(d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {d_is}")
        assert((type(k) == int) and (0 <= k <= self.d))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eta = self.eta
            N = self.N
            lr = group['lr']
            
            for i, (p, m, B) in enumerate(zip(group['params'], group['momentum'], group['scales'])):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                # update the steps for each param group update
                state['step'] += 1

                mu = p.reshape((-1, 1))
                g_bar = p.grad.reshape((-1, 1))
                
                B_inv = B.inv()

                g_mu = eta / N * mu + g_bar
                m.mul_(beta1).add_((1 - beta1) * g_mu)
                
                # Evaluate matrix vector products so intermediate results stay in memory
                update = (B_inv.t() @ (B_inv @ m)).reshape(p.shape)
                p.add_(-lr * update)
                group['scales'][i] = B.update(eta, N, g_bar, beta2=beta2)
        return loss
    
    def ELBO(self, loss_fn):
        def f(preds, labels, M=5, gamma=1):
            """Compute ELBO: L(mu, Sigma) = E_q[sum l_i] - gamma * KL(q || p)
            """
            # When q, p both MVG, we have (https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions)
            # KL(q || p) = 1/2(eta * ||B^{-1}||_F^2 - d + eta * ||mu||^2 - d ln(eta) - 2 ln(det(B))
            # Otherwise, this value needs to be approximated through sampling
            # For the expected value of the summed losses, take M samples and approximate by averaging
            if gamma > 0:
                kl_divergence = -self.d * (1 + np.log(self.eta))
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

