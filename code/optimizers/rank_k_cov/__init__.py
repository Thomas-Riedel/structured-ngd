import numpy as np 
import torch
from matrix_groups.triangular import MUp
from optimizers.noisy_optimizer import *
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class RankCov(NoisyOptimizer):
    def __init__(self, model, N, k=1, lr=1e-3, eta=0.4, damping=0.01, betas=(0.9, 0.999), gamma=1, M=1,
                 weight_decay=0, device='cuda'):
        """

        :param model:
        :param N:
        :param k:
        :param lr:
        :param eta:
        :param damping:
        :param betas:
        :param gamma:
        :param M:
        :param weight_decay:
        :param device:
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.model = model
        params = model.parameters()
        self.N = N
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.damping = damping
        self.eps = 1e-9
        self.device = device
        
        defaults = dict(lr=lr, betas=betas,
                        weight_decay=weight_decay)
        super(NoisyOptimizer, self).__init__(params, defaults)

        d_is = []
        for group in self.param_groups:
            params = group['params']
            group['momentum'], group['scales'] = [], []
            for p in params:
                d_i = int(np.prod(p.shape))
                d_is.append(d_i)
                group['momentum'].append(torch.zeros(d_i, 1, device=self.device))

                # Initialize precision S to prior precision eta / N * I,
                # i.e., B = sqrt(eta / N) * I
                # Note that we cap k to d_i since k \in [0, d_i]
                b_init = np.sqrt(eta / N + damping) * MUp.eye(d_i,
                                                              np.minimum(d_i, k),
                                                              device=self.device)
                group['scales'].append(b_init) # (1)
        self.d = np.sum(d_is)
        print(f"d = {self.d:,}")
        print(f"d_i = {d_is}")
        assert((type(k) == int) and (0 <= k <= self.d))

    @torch.no_grad()
    def sample(self, M):
        """
        Return list z in shape [num_layers, M, *param_dim]
        """
        N = self.N
        params, scales = self.param_groups[0]['params'], self.param_groups[0]['scales']
        z = None
        for p, B in zip(params, scales):
            # z_i ~ N(0, (B B^T)^{-1})
            sample = p.reshape(-1) + torch.rsqrt(torch.tensor(N)) * B.sample(mu=0, n=M).reshape((M, -1))
            if z is None:
                z = sample
                continue
            z = torch.cat((z, sample), dim=1)
        return z

    def step(self, images, labels, loss_fn, closure=None):
        """Performs a single optimization step.

        Args:
            z (nd.array): samples from posterior distribution
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        M = self.M
        z = self.sample(M) # (2)
        params = parameters_to_vector(self.param_groups[0]['params'])
        if closure is None:
            def closure(i):
                self.zero_grad()
                # z_i ~ N(mu, (B B^T)^{-1})
                # Overwrite model weights
                vector_to_parameters(z[i], self.model.parameters())
                # Run forward pass (without sampling again!)
                preds = self.model(images)

                loss = loss_fn(preds, labels)
                loss.backward()
                return loss, preds
        g = []
        preds = None
        loss = 0
        for j in range(M):
            l_i, pred = closure(j)

            loss += l_i
            if preds is None:
                preds = pred
            else:
                preds += pred
            for i, p in enumerate(self.param_groups[0]['params']):
                if p.grad is None:
                    continue
                if j == 0:
                    g_i = torch.zeros((M, np.prod(p.grad.shape), 1), device=self.device)
                    g.append(g_i)
                g[i][j] = p.grad.reshape((-1, 1))
        preds /= M
        loss /= M
        z = z.reshape((z.shape[0], z.shape[1], 1))

        # Return model parameters to original state
        vector_to_parameters(params, self.model.parameters())

        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                beta1, beta2 = group['betas']
                N = self.N
                eta = self.eta
                gamma = self.gamma
                eps = self.eps
                damping = self.damping

                # For each layer
                for i, (p, grad, m, B) in enumerate(zip(group['params'], g, group['momentum'], group['scales'])):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = -1
                    # update the steps for each param group update
                    state['step'] += 1

                    mu = p.reshape((-1, 1))
                    g_bar = torch.mean(grad, axis=0)
                    g_mu = gamma * eta / N * mu + g_bar # (3)
                    m.mul_(beta1).add_((1 - beta1) * g_mu) # (4)

                    t = state['step']
                    m_bar = m / (1 - beta1 ** (t + 1)) # (5)
                    B_bar = torch.rsqrt(torch.tensor(1 - beta2 ** t + eps)) * B.add_id(-np.sqrt(eta / N + damping))
                    B_bar = B_bar.add_id(np.sqrt(eta / N + damping))

                    # Evaluate matrix vector products so intermediate results stay in memory
                    update = (B_bar.inv().t() @ (B_bar.inv() @ m_bar)).reshape(p.shape) # (7)
                    p.add_(-lr * update)

                    group['scales'][i] = B.update(lr, eta, N, grad, z[:, :mu.shape[0]] - mu) # (6) & (8)
                    z = z[:, mu.shape[0]:]
            return loss, preds
    
    def elbo(self, loss_fn):
        def f(preds, labels, M=1, gamma=1):
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
