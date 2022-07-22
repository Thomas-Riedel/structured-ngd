import torch
import seaborn as sns
sns.set()


def h(x):
    """
    Return quadratic approximation to exponential function
        I + X + 1/2 X @ X
    """
    return x.add_id() + 0.5 * x @ x


class MUp:
    def __init__(self, m_a, m_b, m_d, k, device='cuda'):
        assert(m_a.shape[0] == m_a.shape[1])
        assert(m_a.shape[1] == m_b.shape[0])
        assert(m_b.shape[1] == m_d.shape[0])

        self.k = k
        self.d = m_d.shape[0] + k
        self.device = device
        self.m_a = m_a.to(device)
        self.m_b = m_b.to(device)
        self.m_d = m_d.reshape((-1, 1)).to(device)
        self.shape = self.size()
        self.inverse = None
        self.a_inv = None
        self.d_inv = None

    @staticmethod
    def eye(d, k, device='cuda'):
        return MUp(torch.eye(k, k), torch.zeros(k, d-k), torch.ones(d-k), k, device=device)

    @staticmethod
    def zeros(d, k, device='cuda'):
        return MUp(torch.zeros(k, k), torch.zeros(k, d-k), torch.zeros(d-k), k, device=device)

    def to(self, device):
        # Move all blocks onto device
        self.device = device
        self.m_a = self.m_a.to(device)
        self.m_b = self.m_b.to(device)
        self.m_d = self.m_d.reshape((-1, 1)).to(device)
        return self

    def inv(self):
        """
        Calculate inverse of upper triangular block matrix
            ( m_a   m_b )             ( m_a^{-1}   -m_a^{-1} m_b m_d^{-1} )
        M = (           ) => M^{-1} = (                                   )
            ( 0     m_d )             (    0                m_d^{-1}      )
        """
        if self.inverse is None:
            m_a_inv = self.m_a_inv()
            m_d_inv = self.m_d_inv()
            self.inverse = MUp(m_a_inv, -m_a_inv @ (self.m_b * m_d_inv.T), m_d_inv, self.k, device=self.device)
        return self.inverse

    def m_d_inv(self):
        # m_d is diagonal matrix
        if self.d_inv is None:
            self.d_inv = 1/self.m_d
        return self.d_inv

    def m_a_inv(self):
        """
        Calculate inverse for invertible k x k matrix block m_a
        """
        if self.a_inv is None:
            self.a_inv = torch.linalg.inv(self.m_a)
        return self.a_inv

    def full(self):
        """
        Write full d x d matrix into memory as np.array. 
        For large values of d, this will not be able to fit into memory!
        """
        result = torch.zeros((self.d, self.d), device=self.device)
        result[:self.k, :self.k] = self.m_a
        result[:self.k, self.k:] = self.m_b
        result[self.k:, self.k:] = torch.diag(self.m_d)
        return result

    def size(self):
        """
        Return size as torch.Size object
        """
        return torch.Size((self.d, self.d))

    def t(self):
        """
        Return transpose of block matrix which results in block lower triangular matrix
        """
        return MLow(self.m_a.T, self.m_b.T, self.m_d, self.k, device=self.device)

    def precision(self):
        """
        Full precision as arrowhead matrix as np.array (here, we parametrize S = BB^T).
        This ill not fit into memory for large values of d.
        """
        prec = self @ self.t()
        return prec

    def rank_matrix(self):
        """
        Rank k matrix U for which Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2})
        """
        u_k = torch.zeros((self.d, self.k), device=self.device)
        inv = self.m_a_inv()
        u_k[:self.k] = -inv.T
        u_k[self.k:] = self.m_d_inv() * self.m_b.T @ inv.T
        return u_k

    def sigma(self):
        """
        Full low-rank structured covariance matrix Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2})
        Will not fit into memory for large values of d!
        """
        # cov = torch.zeros((self.d, self.d), device=self.device)
        # cov[self.k:, self.k:] = torch.diag(self.m_d_inv().reshape(-1)**2)
        # U_k = self.U()
        # cov += U_k @ U_k.T
        # return cov

        b_inv = self.inv()
        return b_inv.t() @ b_inv

    def corr(self):
        cov = self.sigma()
        std = torch.sqrt(torch.diag(cov)).reshape((-1, 1))
        return 1/std * cov * 1/std.T

    def plot_correlation(self):
        corr = self.corr().cpu()
        sns.heatmap(corr, vmin=-1.0, vmax=1.0, center=0.0, cmap='coolwarm')
      
    def plot_covariance(self):
        cov = self.sigma().cpu()
        sns.heatmap(cov)

    def plot_precision(self):
        prec = self.precision().cpu()
        sns.heatmap(prec)

    def plot_rank_update(self):
        rank_matrix = self.rank_matrix().cpu()
        sns.heatmap(rank_matrix)

    def __repr__(self):
        """
        String representation of class.
        """
        string = "m_a: \n\t"
        string += str(self.m_a)
        string += "\nm_b: \n\t"
        string += str(self.m_b)
        string += "\nm_d: \n\t"
        string += str(self.m_d)
        return string

    def __matmul__(self, x):
        """
        Matrix multiplication M @ X
        """
        assert(self.d == x.shape[0])
        x = x.to(self.device)
        if isinstance(x, MUp) and (self.k == x.k):
            """
            ( m_a1   m_b1 )   ( m_a2   m_b2 )   ( m_a1 @ m_a2   m_a1 @ m_b2 + m_b1 @ m_d2 )
            (             ) @ (             ) = (                                         )
            (  0     m_d1 )   (  0     m_d2 )   (      0               m_d1 @ m_d2        )
            """
            assert(self.k == x.k)
            return MUp(self.m_a @ x.m_a,
                       self.m_a @ x.m_b + self.m_b * x.m_d.T,
                       self.m_d * x.m_d,
                       self.k,
                       device=self.device)
        elif isinstance(x, (int, float)):
            return self * x
        elif isinstance(x, MLow):
            """
            ( m_a1   m_b1 )   ( m_a2     0  )   ( m_a1 @ m_a2 + m_b1 m_c2   m_b1 @ m_d2 )
            (             ) @ (             ) = (                                       )
            (  0     m_d1 )   ( m_c2   m_d2 )   (      m_d1 @ m_c2          m_d1 @ m_d2 )
            """
            assert((self.shape[1] == x.shape[0]) and (self.k == x.k))
            result = torch.zeros((self.shape[0], x.shape[1]), device=self.device)
            result[:self.k, :self.k] = self.m_a @ x.m_a + self.m_b @ x.m_c
            result[:self.k, self.k:] = self.m_b * x.m_d
            result[self.k:, :self.k] = self.m_d * x.m_c
            result[self.k:, self.k:] = torch.diag((self.m_d * x.m_d).reshape(-1))
            return result
        else:
            # x is np.array
            # If x a vector, add one more axis
            if len(x.shape) == 1:
                x = x.reshape((-1, 1))
            result = torch.zeros((self.d, x.shape[1]), device=self.device)
            result[:self.k] = self.m_a @ x[:self.k] + self.m_b @ x[self.k:]
            result[self.k:] = self.m_d * x[self.k:]
            return result
        
    def __rmatmul__(self, x):
        """
        Matrix multiplication X @ M
        """
        assert(x.shape[1] == self.shape[0])
        x = x.to(self.device)
        if isinstance(x, MLow):
            """
            ( m_a1    0   )   ( m_a2   m_b2 )   ( m_a1 @ m_a2   m_a1 @ m_b2               )
            (             ) @ (             ) = (                                         )
            ( m_c1   m_d1 )   (  0     m_d2 )   ( m_c1 @ m_a2   m_c1 @ m_b2 + m_d1 @ m_d2 )
            """
            assert(self.k == x.k)
            result = torch.zeros((x.shape[0], self.shape[1]), device=self.device)
            result[:self.k, :self.k] = x.m_a @ self.m_a
            result[:self.k, self.k:] = x.m_a @ self.m_b
            result[self.k:, :self.k] = x.m_c @ self.m_a
            result[self.k:, self.k:] = x.m_c @ self.m_b + torch.diag((self.m_d * x.m_d).reshape(-1))
            return result
        else:
            # x an np.array
            result = torch.zeros((x.shape[0], self.shape[1])).to(self.device)
            result[:self.k, :self.k] = x[:self.k] @ self.m_a
            result[:self.k, self.k:] = x[:self.k] @ self.m_b + x[:self.k] * self.m_d.T
            result[self.k:, :self.k] = x[self.k:] @ self.m_a

            # This will blow up for large values of d!
            result[self.k:, self.k:] = x[self.k:] * self.m_d.T + x[self.k:] * self.m_d.T

            return result

    def __mul__(self, x):
        """
        (Elementwise) Multiplication X * M
        """
        if isinstance(x, MUp):
            assert(x.k == self.k)
            return MUp(x.m_a * self.m_a,
                       x.m_b * self.m_b,
                       x.m_d * self.m_d,
                       self.k,
                       device=self.device)
        elif isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and (x.ndim == 0)):
            return MUp(x * self.m_a,
                       x * self.m_b,
                       x * self.m_d,
                       self.k,
                       device=self.device)
        elif len(x.shape) == 2:
            # x an np.array
            return MUp(x[:self.k, :self.k] * self.m_a,
                       x[:self.k, self.k:] * self.m_b,
                       torch.diag(x[self.k:, self.k:]).reshape(-1, 1) * self.m_d,
                       self.k,
                       device=self.device)
        else:
            raise ValueError()

    def __rmul__(self, x):
        return self * x

    def __add__(self, x):
        """
          (Elementwise) Addition X + M
        """
        x = x.to(self.device)
        if isinstance(x, MUp):
            assert((self.size() == x.size()) and (self.k == x.k))
            return MUp(x.m_a + self.m_a,
                       x.m_b + self.m_b,
                       x.m_d + self.m_d,
                       self.k,
                       device=self.device)
        elif isinstance(x, (int, float)):
            return MUp(x + self.m_a,
                       x + self.m_b,
                       x + self.m_d,
                       self.k,
                       device=self.device)
        elif len(x.shape) == 2:
            assert(x.shape == self.shape)
            return MUp(x[:self.k, :self.k] + self.m_a,
                       x[:self.k, self.k:] + self.m_b,
                       torch.diag(x[self.k:, self.k:]).reshape(-1, 1) + self.m_d,
                       self.k,
                       device=self.device)
        else:
            raise ValueError()

    def __radd__(self, x):
        return self + x

    def __truediv__(self, x):
        return 1/x * self

    def __neg__(self):
        """Unary minus
        """
        return MUp(-self.m_a, -self.m_b, -self.m_d, self.k, self.device)

    def add_id(self, alpha=1):
        """
        Add alpha * I to M, i.e.,
            self + alpha * I
        """
        return MUp(self.m_a + alpha * torch.eye(self.k),
                   self.m_b,
                   self.m_d + alpha,
                   self.k,
                   device=self.device)

    def trace(self):
        """Compute trace.
            tr(M) = tr(m_a) + tr(m_d) = tr(m_a) + sum(m_d)
        """
        return torch.trace(self.m_a) + torch.sum(self.m_d)

    def det(self):
        """Compute determinant.
            det(M) = det(m_a) * det(m_d) = det(m_a) * (prod m_d)
        """
        return torch.det(self.m_a) * torch.prod(self.m_d)

    def log_det(self):
        """Compute log of determinant.
            log(det(M)) = log(det(m_a) * (prod m_d)) = log(det(m_a)) + sum(log(m_d))
        """
        eps = torch.finfo().tiny
        return torch.logdet(self.m_a + eps * torch.eye(self.k)) + torch.sum(torch.log(self.m_d + eps))

    def frobenius_norm(self):
        """Returns squared Frobenius norm of M, i.e.,
            ||M||_F^2 = tr(M^T M) = ||m_a||_F^2 + ||m_b||_F^2 + ||m_d||_F^2"""
        return torch.sum(self.m_a ** 2) + torch.sum(self.m_b ** 2) + torch.sum(self.m_d ** 2)

    def c_up(self):
        return MUp(0.5 * torch.ones((self.k, self.k)),
                   torch.ones((self.k, self.d - self.k)),
                   0.5 * torch.ones(self.d - self.k),
                   self.k,
                   device=self.device)

    def sample(self, mu=0, n=1):
        """
        Sample z ~ N(mu, Sigma) with covariance matrix 
          Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2}):
        z = mu + U @ eps_rank + diag(0_k, m_d^{-1} @ eps_diag),
          where eps_rank ~ N(0, I_k), eps_diag ~ N(0, I_{d-k})
        Or:
        z = mu + B^{-T} @ eps
          where eps ~ N(0, I_d)
        """
        # eps_rank = torch.randn((self.k, n), device=self.device)
        # eps_diag = torch.randn((self.d - self.k, n), device=self.device)
        # z = mu + (self.U() @ eps_rank + torch.cat((torch.zeros((self.k, n), device=self.device),
        #                                            self.m_d_inv() * eps_diag), 0)).T

        eps = torch.randn((self.d, n), device=self.device)
        z = mu + (self.inv().t() @ eps)
        return z.T

    def update(self, lr, eta, N, g, v, gamma=1):
        """Perform update step 
          B <- B h(lr * C_up .* kappa_up(B^{-1} G_S B^{-T})),
        where h(M) := I + M + 1/2 M^2, kappa_up 'projects' to matrix group B_up
        by zeroing out entries.
        This function however avoids storing intermediate d x d matrices and 
        computes the update step much more efficiently (see algorithm for details).
        """
        assert(gamma >= 0)
        factor = gamma * eta / N
        b_inv = self.inv()
        m_a_inv = b_inv.m_a
        m_d_inv = b_inv.m_d
        AB = m_a_inv @ self.m_b

        if self.k == 0:
            m_a = torch.zeros_like(self.m_a)
            m_b = torch.zeros_like(self.m_b)
            BTv = 0
        else:
            BTv = self.m_b.T @ v[:, :self.k]
            BAT = self.m_b.T @ m_a_inv
            ATv = self.m_a.T @ v[:, :self.k]
            X = g[:, :self.k].transpose(1, 2) @ m_a_inv.T - g[:, self.k:].transpose(1, 2) @ (m_d_inv * BAT)
            M = torch.mean(ATv @ X, axis=0)
            m_a = N/2 * (M + M.T)

            m_b = N/2 * torch.mean(ATv @ (g[:, self.k:] * m_d_inv).transpose(1, 2), axis=0)
            m_b += N/2 * torch.mean(((BTv + self.m_d * v[:, self.k:]) @ X).transpose(1, 2), axis=0)

            if gamma > 0:
                m_a += factor * m_a_inv.T @ m_a_inv - gamma * torch.eye(self.k, device=self.device)
                m_b += -factor * m_a_inv.T @ AB * m_d_inv.T

        m_d = N * torch.mean((v[:, self.k:] + m_d_inv * BTv) * g[:, self.k:], axis=0)
        if gamma > 0:
            m_d += factor * (m_d_inv ** 2) * (1 + torch.sum(AB ** 2, axis=0)).reshape(-1, 1) - gamma
        # print(h(lr * B_up(0.5 * m_a, m_b, 0.5 * m_d, self.k, device=self.device)))

        # We avoid computing C_up * kappa_up(M) by simply multiplying the scalar 
        # values in the respective blocks
        # This returns B @ h(lr * C_up * kappa_up(B^{-1} G_S B^{-T}))
        return self @ h(lr * MUp(0.5 * m_a, m_b, 0.5 * m_d, self.k, device=self.device))


class MLow:
    def __init__(self, m_a, m_c, m_d, k, device='cuda'):
        self.k = k
        self.d = m_d.shape[0] + k
        self.device = device
        self.m_a = m_a.to(device)
        self.m_c = m_c.to(device)
        self.m_d = m_d.to(device).reshape((-1, 1))
        self.shape = self.size()

    @staticmethod
    def eye(d, k, device='cpu'):
        return MLow(torch.eye(k, k), torch.zeros(d-k, k), torch.ones(d-k), k, device=device)

    @staticmethod
    def zeros(d, k, device='cpu'):
        return MLow(torch.zeros(k, k), torch.zeros(d-k, k), torch.zeros(d-k), k, device=device)

    def size(self):
        return torch.Size((self.d, self.d))

    def to(self, device):
        """Copy objects to device
        """
        self.device = device
        self.m_a = self.m_a.to(device)
        self.m_c = self.m_c.to(device)
        self.m_d = self.m_d.reshape((-1, 1)).to(device)
        return self

    def __repr__(self):
        """String representation
        """
        string = "m_a: \n\t"
        string += str(self.m_a)
        string += "\nm_c: \n\t"
        string += str(self.m_c)
        string += "\nm_d: \n\t"
        string += str(self.m_d)
        return string

    def __mul__(self, X):
        if isinstance(X, MUp):
            assert(self.shape == X.shape)
            assert(X.k == self.k)
            return MUp(X.m_a * self.m_a,
                       torch.zeros_like(X.m_b),
                       X.m_d * self.m_d,
                       self.k,
                       device=self.device)
        elif isinstance(X, MLow):
            assert(self.shape == X.shape)
            assert(X.k == self.k)
            return MLow(X.m_a * self.m_a,
                        X.m_c * self.m_c,
                        X.m_d * self.m_d,
                        self.k,
                        device=self.device)
        elif isinstance(X, (int, float)) or (isinstance(X, torch.Tensor) and (X.ndim == 0)):
            return MLow(X * self.m_a,
                        X * self.m_c,
                        X * self.m_d,
                        self.k,
                        device=self.device)
        else:
            # Implement np.array and torch.tensor cases
            raise ValueError()

    def __rmul__(self, x):
        return self * x

    def __matmul__(self, x):
        """Matrix Multiplication B @ x
        """
        assert(self.shape[1] == x.shape[0])

        if isinstance(x, MLow):
            assert(self.k == x.k)
            """
            ( m_a1    0   )   ( m_a2     0  )   ( m_a1 @ m_a2                      0      )
            (             ) @ (             ) = (                                         )
            ( m_c1   m_d1 )   ( m_c2   m_d2 )   ( m_c1 @ m_a2 + m_d1 @ m_c2   m_d1 @ m_d2 )
            """
            return MLow(self.m_a @ x.m_a,
                        self.m_c @ x.m_a + self.m_d * x.m_c,
                        self.m_d * x.m_d,
                        self.k,
                        device=self.device)
        if isinstance(x, MUp):
            assert((self.shape[1] == x.shape[0]) and (self.k == x.k))
            """
            ( m_a1    0   )   ( m_a2   m_b2 )   ( m_a1 @ m_a2   m_a1 @ m_b2               )
            (             ) @ (             ) = (                                         )
            ( m_c1   m_d1 )   (  0     m_d2 )   ( m_c1 @ m_a2   m_c1 @ m_b2 + m_d1 @ m_d2 )
            """
            assert(self.k == x.k)
            result = torch.zeros((x.shape[0], self.shape[1]), device=self.device)
            result[:self.k, :self.k] = self.m_a @ x.m_a
            result[:self.k, self.k:] = self.m_a @ x.m_b
            result[self.k:, :self.k] = self.m_c @ x.m_a
            result[self.k:, self.k:] = self.m_c @ x.m_b + torch.diag((self.m_d * x.m_d).reshape(-1))
            return result
        elif isinstance(x, (int, float)):
            return self * x
        else:
            # Write as block matrix!
            if len(x.shape) == 1:
                result = torch.zeros((self.d, 1), device=self.device)
            else:
                result = torch.zeros((self.d, x.shape[1]), device=self.device)
            result[:self.k] = self.m_a @ x[:self.k]
            result[self.k:] = self.m_c @ x[:self.k] + self.m_d * x[self.k:]
            return result
