import numpy as np
import torch
import seaborn as sns
sns.set()


def h(M):
    return B_up.eye(M.d, M.k, device=M.device) + M + 0.5 * M @ M


class B_up:
    def __init__(self, B_a, B_b, B_d, k, device='cuda'):
        self.k = k
        self.d = B_d.shape[0] + k
        self.device = device
        self.B_a = B_a.to(device)
        self.B_b = B_b.to(device)
        self.B_d = B_d.reshape((-1, 1)).to(device)
        self.shape = self.size()
        assert(B_a.shape[0] == B_a.shape[1])
        assert(B_a.shape[1] == B_b.shape[0])
        assert(B_b.shape[1] == B_d.shape[0])

    @staticmethod
    def eye(d, k, device='cuda'):
        return B_up(torch.eye(k, k), torch.zeros(k, d-k), torch.ones(d-k), k, device=device)

    @staticmethod
    def zeros(d, k, device='cuda'):
        return B_up(torch.zeros(k, k), torch.zeros(k, d-k), torch.zeros(d-k), k, device=device)

    def to(self, device):
        # Move all blocks onto device
        self.device = device
        self.B_a = self.B_a.to(device)
        self.B_b = self.B_b.to(device)
        self.B_d = self.B_d.reshape((-1, 1)).to(device)
        return self

    def inv(self):
        """
        Calculate inverse of upper triangular block matrix
            ( B_a   B_b )             ( B_a^{-1}   -B_a^{-1} B_b B_d^{-1} )
        B = (           ) => B^{-1} = (                                   )
            ( 0     B_d )             (    0                B_d^{-1}      )
        """
        B_a_inv = self.B_a_inv()
        B_d_inv = self.B_d_inv()
        return B_up(B_a_inv, -B_a_inv @ (self.B_b * B_d_inv.T), B_d_inv, self.k, device=self.device)

    def B_d_inv(self):
        # B_d is diagonal matrix
        return 1/self.B_d

    def B_a_inv(self):
        """
        Calculate inverse for invertible k x k matrix block B_a
        """
        return torch.linalg.inv(self.B_a)

    def full(self):
        """
        Write full d x d matrix into memory as np.array. 
        For large values of d, this will not be able to fit into memory!
        """
        B = torch.zeros((self.d, self.d), device=self.device)
        B[:self.k, :self.k] = self.B_a
        B[:self.k, self.k:] = self.B_b
        B[self.k:, self.k:] = torch.diag(self.B_d)
        return B

    def size(self):
        """
        Return size as torch.Size object
        """
        return torch.Size((self.d, self.d))

    def t(self):
        """
        Return transpose of block matrix which results in block lower triangular matrix
        """
        return B_low(self.B_a.T, self.B_b.T, self.B_d, self.k, device=self.device)

    def S(self):
        """
        Full precision as arrowhead matrix as an np.array (here, we parametrize S = BB^T).
        This ill not fit into memory for large values of d.
        """
        prec = self @ self.t()
        return prec

    def U(self):
        """
        Rank k matrix U for which Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, B_d^{-2})
        """
        U_k = torch.zeros((self.d, self.k), device=self.device)
        inv = self.B_a_inv()
        U_k[:self.k] = -inv.T
        U_k[self.k:] = self.B_d_inv() * self.B_b.T @ inv.T
        return U_k

    def Sigma(self):
        """
        Full low-rank structured covariance matrix Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, B_d^{-2})
        Will not fit into memory for large values of d!
        """
        # cov = torch.zeros((self.d, self.d), device=self.device)
        # cov[self.k:, self.k:] = torch.diag(self.B_d_inv().reshape(-1)**2)
        # U_k = self.U()
        # cov += U_k @ U_k.T
        # return cov

        B_inv = self.inv()
        return B_inv.t() @ B_inv

    def corr(self):
        cov = self.Sigma()
        std = torch.sqrt(torch.diag(cov)).reshape((-1, 1))
        return 1/std * cov * 1/std.T

    def plot_correlation(self):
        corr = self.corr().cpu()
        sns.heatmap(corr, vmin=-1.0, vmax=1.0, center=0.0, cmap='coolwarm')
      
    def plot_covariance(self):
        cov = self.Sigma().cpu()
        sns.heatmap(cov)

    def plot_precision(self):
        prec = self.S().cpu()
        sns.heatmap(prec)

    def plot_rank_update(self):
        U = self.U().cpu()
        sns.heatmap(U)

    def __repr__(self):
        """
        String representation of class.
        """
        string = "B_a: \n\t"
        string += str(self.B_a)
        string += "\nB_b: \n\t"
        string += str(self.B_b)
        string += "\nB_d: \n\t"
        string += str(self.B_d)
        return string

    def __matmul__(self, X):
        """
        Matrix multiplication B @ X
        """
        assert(self.d == X.shape[0])
        X = X.to(self.device)
        if isinstance(X, B_up) and (self.k == X.k):
            """
            ( B_a1   B_b1 )   ( B_a2   B_b2 )   ( B_a1 @ B_a2   B_a1 @ B_b2 + B_b1 @ B_d2 )
            (             ) @ (             ) = (                                         )
            (  0     B_d1 )   (  0     B_d2 )   (      0               B_d1 @ B_d2        )
            """
            assert(self.k == X.k)
            return B_up(self.B_a @ X.B_a, 
                        self.B_a @ X.B_b + self.B_b * X.B_d.T, 
                        self.B_d * X.B_d, self.k, 
                        device=self.device)
        elif isinstance(X, (int, float)):
            return self * X
        elif isinstance(X, B_low):
            """
            ( B_a1   B_b1 )   ( B_a2     0  )   ( B_a1 @ B_a2 + B_b1 B_c2   B_b1 @ B_d2 )
            (             ) @ (             ) = (                                       )
            (  0     B_d1 )   ( B_c2   B_d2 )   (      B_d1 @ B_c2          B_d1 @ B_d2 )
            """
            assert((self.shape[1] == X.shape[0]) and (self.k == X.k))
            result = torch.zeros((self.shape[0], X.shape[1]), device=self.device)
            result[:self.k, :self.k] = self.B_a @ X.B_a + self.B_b @ X.B_c
            result[:self.k, self.k:] = self.B_b * X.B_d
            result[self.k:, :self.k] = self.B_d * X.B_c
            result[self.k:, self.k:] = torch.diag((self.B_d * X.B_d).reshape(-1))
            return result
        else:
            # X an np.array
            # If X is vector, add one more axis
            if len(X.shape) == 1:
                X = X.reshape((-1, 1))
            result = torch.zeros((self.d, X.shape[1]), device=self.device)
            result[:self.k] = self.B_a @ X[:self.k] + self.B_b @ X[self.k:]
            result[self.k:] = self.B_d * X[self.k:]
            return result
        
    def __rmatmul__(self, X):
        """
        Matrix multiplication X @ B
        """
        assert(X.shape[1] == self.shape[0])
        X = X.to(self.device)
        if isinstance(X, B_low):
            """
            ( B_a1    0   )   ( B_a2   B_b2 )   ( B_a1 @ B_a2   B_a1 @ B_b2               )
            (             ) @ (             ) = (                                         )
            ( B_c1   B_d1 )   (  0     B_d2 )   ( B_c1 @ B_a2   B_c1 @ B_b2 + B_d1 @ B_d2 )
            """
            assert(self.k == X.k)
            result = torch.zeros((X.shape[0], self.shape[1]), device=self.device)
            result[:self.k, :self.k] = X.B_a @ self.B_a
            result[:self.k, self.k:] = X.B_a @ self.B_b
            result[self.k:, :self.k] = X.B_c @ self.B_a
            result[self.k:, self.k:] = X.B_c @ self.B_b + torch.diag((self.B_d * X.B_d).reshape(-1))
            return result
        else:
            # X an np.array
            result = torch.zeros((X.shape[0], self.shape[1])).to(self.device)
            result[:self.k, :self.k] = X[:self.k] @ self.B_a
            result[:self.k, self.k:] = X[:self.k] @ self.B_b + X[:self.k] * self.B_d.T
            result[self.k:, :self.k] = X[self.k:] @ self.B_a

            # This will blow up for large values of d!
            result[self.k:, self.k:] = X[self.k:] * self.B_d.T + X[self.k:] * self.B_d.T

            return result

    def __mul__(self, X):
        """
        (Elementwise) Multiplication X * B
        """
        if isinstance(X, B_up):
            assert(X.k == self.k)
            return B_up(X.B_a * self.B_a, X.B_b * self.B_b, X.B_d * self.B_d, self.k, device=self.device)
        elif isinstance(X, (int, float)):
            return B_up(X * self.B_a, X * self.B_b, X * self.B_d, self.k, device=self.device)
        elif len(X.shape) == 2:
            # X is np.array
            return B_up(X[:self.k, :self.k] * self.B_a, 
                        X[:self.k, self.k:] * self.B_b, 
                        torch.diag(X[self.k:, self.k:]).reshape(-1, 1) * self.B_d, 
                        self.k, 
                        device=self.device)
        else:
            raise ValueError()

    def __rmul__(self, x):
        return self * x

    def __add__(self, X):
        """
          (Elementwise) Addition X + B
        """
        X = X.to(self.device)
        if isinstance(X, B_up):
            assert((self.size() == X.size()) and (self.k == X.k))
            return B_up(X.B_a + self.B_a, X.B_b + self.B_b, X.B_d + self.B_d, 
                        self.k, device=self.device)
        elif isinstance(X, (int, float)):
            return B_up(X + self.B_a, X + self.B_b, X + self.B_d, self.k, 
                        device=self.device)
        elif len(X.shape) == 2:
            assert(X.shape == self.shape)
            return B_up(X[:self.k, :self.k] + self.B_a, 
                        X[:self.k, self.k:] + self.B_b, 
                        torch.diag(X[self.k:, self.k:]).reshape(-1, 1) + self.B_d, 
                        self.k, 
                        device=self.device)
        else:
            raise ValueError()

    def __radd__(self, X):
        return self + X

    def __neg__(self):
        """Unary minus
        """
        return B_up(-self.B_a, -self.B_b, -self.B_d, self.k, self.device)

    def trace(self):
        """Compute trace.
        trace(B) = trace(B_a) + trace(B_d) = trace(B_a) + sum(B_d)
        """
        return torch.trace(self.B_a) + torch.sum(self.B_d)

    def det(self):
        """Compute determinant.
        det(B) = det(B_a) * det(B_d) = det(B_a) * (prod B_d)
        """
        return torch.det(self.B_a) * torch.prod(self.B_d)

    def log_det(self):
        """Compute log of determinant.
        log(det(B)) = log(det(B_a) * (prod B_d)) = log(det(B_a)) + sum(log(B_d))
        """
        return torch.log(torch.det(self.B_a)) + torch.sum(torch.log(self.B_d))

    def frobenius_norm(self):
        """Returns squared Frobenius norm of B, i.e., ||B||_F^2 = tr(B^T B)"""
        return torch.sum(self.B_a ** 2) + torch.sum(self.B_b ** 2) + torch.sum(self.B_d ** 2)

    def C_up(self):
        return B_up(0.5 * torch.ones((self.k, self.k)), 
                    torch.ones((self.k, self.d - self.k)), 
                    0.5 * torch.ones(self.d - self.k), 
                    self.k,
                    device=self.device)

    def sample(self, mu=0, n=1):
        """
        Sample z ~ N(mu, Sigma) with covariance matrix 
          Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, B_d^{-2}):
        z = mu + U @ eps_rank + diag(0_k, B_d^{-1} @ eps_diag), 
          where eps_rank ~ N(0, I_k), eps_diag ~ N(0, I_{d-k})
        Or:
        z = mu + B^{-T} @ eps
          where eps ~ N(0, I_d)
        """
        # eps_rank = torch.randn((self.k, n), device=self.device)
        # eps_diag = torch.randn((self.d - self.k, n), device=self.device)
        # z = mu + (self.U() @ eps_rank + torch.cat((torch.zeros((self.k, n), device=self.device),
        #                                            self.B_d_inv() * eps_diag), 0)).T

        eps = torch.randn((self.d, n), device=self.device)
        z = mu + (self.inv().t() @ eps)
        return z.T

    def update(self, eta, N, g, beta2=0.999):
        """Perform update step 
          B <- B h((1 - beta2) * C_up .* kappa_up(B^{-T} G_S B^{-1})),
        where h(M) := I + M + 1/2 M^2, kappa_up 'projects' to matrix group B_up
        by zeroing out entries.
        This function however avoids storing intermediate d x d matrices and 
        computes the update step much more efficiently (see algorithm for details).
        """

        B_inv = self.inv()
        B_a_inv = B_inv.B_a
        B_d_inv = B_inv.B_d

        # Sample v ~ N(0, 1/N * B^{-T} B^{-1}) = N(0, (sqrt(N) B)^{-T} (sqrt(N) B)^{-1})
        eps = torch.randn((self.d, 1), device=self.device)
        v = 1/np.sqrt(N) * B_inv.t() @ eps

        AB = B_a_inv @ self.B_b
        BAT = self.B_b.T @ B_a_inv
        ATv = self.B_a.T @ v[:self.k]
        BTv = self.B_b.T @ v[:self.k]
        X = g[:self.k].T @ B_a_inv.T - g[self.k:].T @ (B_d_inv * BAT)
        
        B_a = eta / N * B_a_inv.T @ B_a_inv
        B_a += -torch.eye(self.k, device=self.device)
        M = ATv @ X
        B_a += N/2 * (M + M.T)

        B_b = -eta / N * B_a_inv.T @ AB * B_d_inv.T
        B_b += N/2 * ATv @ (g[self.k:] * B_d_inv).T
        B_b += (N/2 * (BTv + self.B_d * v[self.k:]) @ X).T

        B_d = eta / N * (B_d_inv ** 2) * (1 + torch.sum(AB ** 2, axis=0)).reshape(-1, 1)
        B_d += -1
        B_d += N * (v[self.k:] + B_d_inv * BTv) * g[self.k:]

        # We avoid computing C_up * kappa_up(M) by simply multiplying the scalar 
        # values in the respective blocks
        # This returns B @ h((1 - beta2) * C_up * kappa_up(B^{-T} G_S B^{-1}))
        return self @ h((1 - beta2) * B_up(0.5 * B_a, B_b, 0.5 * B_d, self.k, device=self.device))


class B_low:
    def __init__(self, B_a, B_c, B_d, k, device='cuda'):
        self.k = k
        self.d = B_d.shape[0] + k
        self.device = device
        self.B_a = B_a.to(device)
        self.B_c = B_c.to(device)
        self.B_d = B_d.to(device).reshape((-1, 1))
        self.shape = self.size()

    @staticmethod
    def eye(d, k, device='cpu'):
        return B_low(torch.eye(k, k), torch.zeros(d-k, k), torch.ones(d-k), k, device=device)

    @staticmethod
    def zeros(d, k, device='cpu'):
        return B_low(torch.zeros(k, k), torch.zeros(d-k, k), torch.zeros(d-k), k, device=device)

    def size(self):
        return torch.Size((self.d, self.d))

    def to(self, device):
        """Copy objects to device
        """
        self.device = device
        self.B_a = self.B_a.to(device)
        self.B_c = self.B_c.to(device)
        self.B_d = self.B_d.reshape((-1, 1)).to(device)
        return self

    def __repr__(self):
        """String representation
        """
        string = "B_a: \n\t"
        string += str(self.B_a)
        string += "\nB_c: \n\t"
        string += str(self.B_c)
        string += "\nB_d: \n\t"
        string += str(self.B_d)
        return string

    def __mul__(self, X):
        if isinstance(X, B_up):
            assert(self.shape == X.shape)
            assert(X.k == self.k)
            return B_up(X.B_a * self.B_a, 
                        torch.zeros_like(X.B_b), 
                        X.B_d * self.B_d, 
                        self.k, 
                        device=self.device)
        elif isinstance(X, B_low):
            assert(self.shape == X.shape)
            assert(X.k == self.k)
            return B_low(X.B_a * self.B_a, 
                         X.B_c * self.B_c, 
                         X.B_d * self.B_d, 
                         self.k, 
                         device=self.device)
        elif isinstance(X, (int, float)):
            return B_low(X * self.B_a, 
                         X * self.B_c, 
                         X * self.B_d, 
                         self.k, 
                         device=self.device)
        else:
            # Implement np.array and torch.tensor cases
            raise ValueError()

    def __rmul__(self, X):
        return self * X

    def __matmul__(self, X):
        """Matrix Multiplication B @ X
        """
        assert(self.shape[1] == X.shape[0])

        if isinstance(X, B_low):
            assert(self.k == X.k)
            """
            ( B_a1    0   )   ( B_a2     0  )   ( B_a1 @ B_a2                      0      )
            (             ) @ (             ) = (                                         )
            ( B_c1   B_d1 )   ( B_c2   B_d2 )   ( B_c1 @ B_a2 + B_d1 @ B_c2   B_d1 @ B_d2 )
            """
            return B_low(self.B_a @ X.B_a, 
                         self.B_c @ X.B_a + self.B_d * X.B_c, 
                         self.B_d * X.B_d, 
                         self.k, 
                         device=self.device)
        if isinstance(X, B_up):
            assert((self.shape[1] == X.shape[0]) and (self.k == X.k))
            """
            ( B_a1    0   )   ( B_a2   B_b2 )   ( B_a1 @ B_a2   B_a1 @ B_b2               )
            (             ) @ (             ) = (                                         )
            ( B_c1   B_d1 )   (  0     B_d2 )   ( B_c1 @ B_a2   B_c1 @ B_b2 + B_d1 @ B_d2 )
            """
            assert(self.k == X.k)
            result = torch.zeros((X.shape[0], self.shape[1]), device=self.device)
            result[:self.k, :self.k] = self.B_a @ X.B_a
            result[:self.k, self.k:] = self.B_a @ X.B_b
            result[self.k:, :self.k] = self.B_c @ X.B_a
            result[self.k:, self.k:] = self.B_c @ X.B_b + torch.diag((self.B_d * X.B_d).reshape(-1))
            return result
        elif isinstance(X, (int, float)):
            return self * X
        else:
            # Write as block matrix!
            if len(X.shape) == 1:
                result = torch.zeros((self.d, 1), device=self.device)
            else:
                result = torch.zeros((self.d, X.shape[1]), device=self.device)
            result[:self.k] = self.B_a @ X[:self.k]
            result[self.k:] = self.B_c @ X[:self.k] + self.B_d * X[self.k:]
            return result
