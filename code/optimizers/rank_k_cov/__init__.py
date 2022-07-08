import numpy as np 
import torch
from torch.optim import Optimizer
from matrix_groups.triangular import B_up


class Rank_kCov(Optimizer):
    def __init__(self, params, N, k=1, lr=1e-1, eta=0.4, damping=1, betas=(0.9, 0.999),
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
        self.damping = damping
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
                group['momentum'].append(torch.zeros(d_i, 1, device=self.device))

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                B_init = np.sqrt(eta / N + damping) * B_up.eye(d_i,
                                                     np.minimum(d_i, k),
                                                     device=self.device)
                group['scales'].append(B_init)
        self.d = np.sum(d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {d_is}")
        assert((type(k) == int) and (0 <= k <= self.d))

    def sample(self, M):
        N = self.N
        params, scales = self.param_groups[0]['params'], self.param_groups[0]['scales']
        z = None
        for p, B in zip(params, scales):
            # z_i ~ N(0, (B B^T)^{-1})
            sample = p.reshape(-1) + 1/np.sqrt(N) * B.sample(mu=0, n=M).reshape((M, -1))
            if z is None:
                z = sample
                continue
            z = torch.cat((z, sample), dim=1)
        return z

    def step(self, z, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        g = []
        loss = 0
        for j in range(len(z)):
            loss += closure(j)
            for i, p in enumerate(self.param_groups[0]['params']):
                if p.grad is None:
                    continue
                if j == 0:
                    g_i = torch.zeros((len(z), np.prod(p.grad.shape), 1), device=self.device)
                    g.append(g_i)
                g[i][j] = p.grad.reshape((-1, 1))
        loss /= len(z)
        z = z.reshape((len(z), z.shape[1], 1))

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                N = self.N
                eta = self.eta
                damping = self.damping

                for i, (p, grad, m, B) in enumerate(zip(group['params'], g, group['momentum'], group['scales'])):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                    # update the steps for each param group update
                    state['step'] += 1

                    mu = p.reshape((-1, 1))
                    g_bar = torch.mean(grad, axis=0)
                    g_mu = eta / N * mu + g_bar
                    m.mul_(beta1).add_((1 - beta1) * g_mu)

                    m_bar = m / (1 - beta1 ** state['step'])
                    B_inv_bar = (B / np.sqrt(1 - beta2 ** state['step'])).add_I(np.sqrt(damping)).inv()

                    # Evaluate matrix vector products so intermediate results stay in memory
                    update = (B_inv_bar.t() @ (B_inv_bar @ m_bar)).reshape(p.shape)
                    p.add_(-lr * update)
                    group['scales'][i] = B.update(eta, N, grad, z[:, :mu.shape[0]] - mu, beta2=beta2)
                    z = z[:, mu.shape[0]:]
            return loss
    
    def ELBO(self, loss_fn):
        def f(preds, labels, M=1, gamma=1):
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
