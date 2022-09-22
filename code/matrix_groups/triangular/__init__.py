import numpy as np
import torch
from typing import Union, List


def solve(A, b):
    eps = 1e-12
    norm = torch.linalg.norm(b) + eps
    return norm * torch.linalg.lstsq(A, b / norm).solution

class MUp:
    def __init__(self, m_a: np.array, m_b: np.array, m_d: np.array, k: int,
                 device: str = None, damping: float = 0.0) -> None:
        """Block upper triangular matrix class
                ( m_a   m_b )
            M = (           )
                ( 0     m_d )

        :param m_a: np.array of shape (k, k), first block
        :param m_b: np.array of shape (k, d-k), second block
        :param m_d: np.array of shape (d-k), diagonal third block
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        """
        assert(m_a.shape[0] == m_a.shape[1])
        assert(m_a.shape[1] == m_b.shape[0])
        assert(m_b.shape[1] == m_d.shape[0])
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.k = k
        self.d = m_d.shape[0] + k
        self.device = device
        self.m_a = m_a.to(device)
        self.m_b = m_b.to(device)
        self.m_d = m_d.reshape((-1, 1)).to(device)
        self.shape = self.size()
        self.damping = damping
        self.inverse = None
        self.a_inv = None
        self.d_inv = None

    @staticmethod
    def eye(d: int, k: int, device: str = None, damping: float = 0.0):
        """Identity matrix of shape d x d represented as MUp.

        :param d: int, size of matrix
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        :return: result: MUp, identity matrix represented as MUp
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return MUp(torch.eye(k, k), torch.zeros(k, d-k), torch.ones(d-k), k, device=device, damping=damping)

    @staticmethod
    def zeros(d: int, k: int, device: str = None, damping: float = 0.0):
        """Zero matrix of shape d x d represented as MUp.

        :param d: int, size of matrix
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        :return: result: MUp, zero matrix represented as MUp
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return MUp(torch.zeros(k, k), torch.zeros(k, d-k), torch.zeros(d-k), k, device=device, damping=damping)

    def to(self, device: str):
        """Move object onto device.
        
        :param device: str, torch device to run operations on (GPU or CPU)
        :return: self, MUp matrix on specified device
        """
        # Move all blocks onto device
        self.device = device
        self.m_a = self.m_a.to(device)
        self.m_b = self.m_b.to(device)
        self.m_d = self.m_d.reshape((-1, 1)).to(device)
        return self

    def solve(self, b: np.array) -> np.array:
        """Solve (B + damping * I) x = b
        
        :param b: np.array of shape (d, 1), Right hand side of linear system of equations
        :return: result, np.array of shape (d, 1) as solution of dampened linear system
        """
        assert(b.shape[0] == self.d)
        # Thomas algorithm, see self.inv() for information 
        identity = torch.eye(self.k, device=self.device)
        m_a = self.m_a + self.damping * identity
        m_d = self.m_d + self.damping
        result = torch.zeros_like(b, device=self.device)
        result[self.k:] = b[self.k:] / m_d
        result[:self.k] = solve(m_a, b[:self.k] - self.m_b @ result[self.k:])
        return result

    def inv(self):
        """Calculate inverse of upper triangular block matrix 
                ( m_a   m_b )             ( m_a^{-1}   -m_a^{-1} m_b m_d^{-1} )
            M = (           ) => M^{-1} = (                                   )
                ( 0     m_d )             (    0                m_d^{-1}      )
        :return: self.inverse, MUp representation of inverse
        """
        if self.inverse is None:
            m_a_inv = self.m_a_inv()
            m_d_inv = self.m_d_inv()
            self.inverse = MUp(m_a_inv,
                               -m_a_inv @ (self.m_b * m_d_inv.T),
                               m_d_inv,
                               self.k,
                               device=self.device,
                               damping=self.damping)
        return self.inverse

    def m_a_inv(self) -> np.array:
        """Calculate dampened inverse for invertible k x k matrix block m_a
        
        :return: self.a_inv, torch.Tensor, dampened inverse of first block
        """
        if self.a_inv is None:
            identity = torch.eye(self.k, device=self.device)
            self.a_inv = solve(self.m_a + self.damping * identity, identity)
        return self.a_inv

    def m_d_inv(self) -> np.array:
        """Compute dampened inverse of diagonal block.
        
        :return: self.d_inv, torch.Tensor, dampened inverse of diagonal block
        """
        # m_d is diagonal matrix
        if self.d_inv is None:
            self.d_inv = 1/(self.m_d + self.damping)
        return self.d_inv

    def full(self) -> np.array:
        """Write full d x d matrix into memory as np.array. 
        For large values of d, this will not be able to fit into memory!
        
        :return: result, np.array, dense matrix
        """
        result = torch.zeros((self.d, self.d), device=self.device)
        result[:self.k, :self.k] = self.m_a
        result[:self.k, self.k:] = self.m_b
        result[self.k:, self.k:] = torch.diag(self.m_d)
        return result

    def size(self):
        """Return size as torch.Size object.
        
        :return: size, torch.Size, (d, d) tuple specifying size
        """
        return torch.Size((self.d, self.d))

    def t(self):
        """Return transpose of block matrix which results in block lower triangular matrix.
        
        :return: transpose, MLow, transpose of object
        """
        return MLow(self.m_a.T, self.m_b.T, self.m_d, self.k, device=self.device, damping=self.damping)

    def precision(self) -> np.array:
        """Full precision as arrowhead matrix as np.array (here, we parametrize S = B @ B^T).
        This will not fit into memory for large values of d.
        
        :return: precision, np.array, full representation of precision matrix as B @ B^T
        """
        precision = self @ self.t()
        return precision

    def rank_matrix(self) -> np.array:
        """Rank k matrix U for which Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2})

        :return: u_k, np.array, rank k matrix
        """
        u_k = torch.zeros((self.d, self.k), device=self.device)
        inv = self.m_a_inv()
        u_k[:self.k] = -inv.T
        u_k[self.k:] = self.m_d_inv() * self.m_b.T @ inv.T
        return u_k

    def sigma(self) -> np.array:
        """Full low-rank structured covariance matrix Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2})
        Will not fit into memory for large values of d!

        :return: sigma, np.array, dense representation of covariance matrix
        """
        # cov = torch.zeros((self.d, self.d), device=self.device)
        # cov[self.k:, self.k:] = torch.diag(self.m_d_inv().reshape(-1)**2)
        # U_k = self.U()
        # cov += U_k @ U_k.T
        # return cov

        b_inv = self.inv()
        return b_inv.t() @ b_inv

    def corr(self) -> np.array:
        """Normalized correlation matrix

        :return: corr, np.array, normalized correlation matrix
        """
        cov = self.sigma()
        std = torch.sqrt(torch.diag(cov)).reshape((-1, 1))
        return 1/std * cov * 1/std.T

    def plot_correlation(self) -> None:
        """Plot heatmap of correlation matrix.
        """
        import seaborn as sns
        sns.set()

        corr = self.corr().cpu()
        sns.heatmap(corr, vmin=-1.0, vmax=1.0, center=0.0, cmap='coolwarm')
      
    def plot_covariance(self) -> None:
        """Plot heatmap of covariance matrix.
        """
        import seaborn as sns
        sns.set()

        cov = self.sigma().cpu()
        sns.heatmap(cov)

    def plot_precision(self) -> None:
        """Plot heatmap of precision matrix.
        """
        import seaborn as sns
        sns.set()

        prec = self.precision().cpu()
        sns.heatmap(prec)

    def plot_rank_update(self) -> None:
        """Plot eatmap of low rank update.
        """
        import seaborn as sns
        sns.set()

        rank_matrix = self.rank_matrix().cpu()
        sns.heatmap(rank_matrix)

    def __repr__(self) -> str:
        """String representation of class including blocks: m_a, m_b, m_d

        :return: string, str, string representation of class
        """
        string = "m_a: \n\t"
        string += str(self.m_a)
        string += "\nm_b: \n\t"
        string += str(self.m_b)
        string += "\nm_d: \n\t"
        string += str(self.m_d)
        return string

    def __matmul__(self, x: Union[int, float, np.array]) -> np.array:
        """Matrix multiplication M @ X

        :param x: Union[int, float, np.array], scalar, vector, or matrix for matrix multiplication
        :return: result, result of matrix multiplication with B
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
                       device=self.device,
                       damping=self.damping)
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
        
    def __rmatmul__(self, x: Union[int, float, np.array]) -> np.array:
        """Matrix multiplication X @ M

        :param x: Union[int, float, np.array], scalar, vector, or matrix for matrix multiplication
        :return: result, result of matrix multiplication with B
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

    def __mul__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Multiplication X * M

        :param x: Union[int, float, np.array, MUp], argument for elementwise multiplication; for scalar,
            broadcasted multiplication
        :return: result, MUp, result of multiplication
        """
        if isinstance(x, MUp):
            assert(x.k == self.k)
            return MUp(x.m_a * self.m_a,
                       x.m_b * self.m_b,
                       x.m_d * self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        elif isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and (x.ndim == 0)):
            return MUp(x * self.m_a,
                       x * self.m_b,
                       x * self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        elif len(x.shape) == 2:
            # x an np.array
            return MUp(x[:self.k, :self.k] * self.m_a,
                       x[:self.k, self.k:] * self.m_b,
                       torch.diag(x[self.k:, self.k:]).reshape(-1, 1) * self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        else:
            raise ValueError()

    def __rmul__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Multiplication X * M

        :param x: Union[int, float, np.array, MUp], argument for elementwise multiplication; for scalar,
            broadcasted multiplication
        :return: result, MUp, result of multiplication
        """
        return self * x

    def __add__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Addition X + M

        :param x: Union[int, float, np.array, MUp], argument for elementwise addition; for scalar,
            broadcasted addition
        :return: result, MUp, result of addition
        """
        x = x.to(self.device)
        if isinstance(x, MUp):
            assert((self.size() == x.size()) and (self.k == x.k))
            return MUp(x.m_a + self.m_a,
                       x.m_b + self.m_b,
                       x.m_d + self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        elif isinstance(x, (int, float)):
            return MUp(x + self.m_a,
                       x + self.m_b,
                       x + self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        elif len(x.shape) == 2:
            assert(x.shape == self.shape)
            return MUp(x[:self.k, :self.k] + self.m_a,
                       x[:self.k, self.k:] + self.m_b,
                       torch.diag(x[self.k:, self.k:]).reshape(-1, 1) + self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        else:
            raise ValueError()

    def __radd__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Addition X + M

        :param x: Union[int, float, np.array, MUp], argument for elementwise addition; for scalar,
            broadcasted addition
        :return: result, MUp, result of addition
        """
        return self + x

    def __truediv__(self, x: Union[int, float]):
        """(Elementwise) Division M / x

        :param x: Union[int, float, np.array, MUp], argument for elementwise division
        :return: result, MUp, result of division
        """
        return 1/x * self

    def __neg__(self):
        """Unary minus.

        :return: result, MUp, negated entries
        """
        return MUp(-self.m_a, -self.m_b, -self.m_d, self.k, self.device, damping=self.damping)

    def add_id(self, alpha: float = 1):
        """Add alpha * I to M, i.e.,
                            ( m_a1 + alpha * I           m_b1     )
            M + alpha * I = (                                     )
                            (  0                 m_d1 + alpha * I )

        :param alpha: float, factor to identity matrix to be added to M
        :return: result, MUp, result of M + alpha * I
        """
        return MUp(self.m_a + alpha * torch.eye(self.k, device=self.device),
                   self.m_b,
                   self.m_d + alpha,
                   self.k,
                   device=self.device,
                   damping=self.damping)

    def trace(self) -> float:
        """Compute trace.
            tr(M) = tr(m_a) + tr(m_d) = tr(m_a) + sum(m_d)

        :return: trace, float, trace of M
        """
        return torch.trace(self.m_a) + torch.sum(self.m_d)

    def det(self) -> float:
        """Compute determinant.
            det(M) = det(m_a) * det(m_d) = det(m_a) * (prod m_d)

        :return: det, float, result of determinant of M
        """
        return torch.det(self.m_a) * torch.prod(self.m_d)

    def log_det(self) -> float:
        """Compute log of determinant.
            log(det(M)) = log(det(m_a) * (prod m_d)) = log(det(m_a)) + sum(log(m_d))

        :return: log_det, float, result of log(det(M))
        """
        eps = torch.finfo().tiny
        return torch.logdet(self.m_a + eps * torch.eye(self.k)) + torch.sum(torch.log(self.m_d + eps))

    def frobenius_norm(self) -> float:
        """Returns squared Frobenius norm of M, i.e.,
            ||M||_F^2 = tr(M^T M) = ||m_a||_F^2 + ||m_b||_F^2 + ||m_d||_F^2

        :return: norm, float, squared Frobenius norm of M
        """
        return torch.sum(self.m_a ** 2) + torch.sum(self.m_b ** 2) + torch.sum(self.m_d ** 2)

    def c_up(self):
        """C_up for Natural Gradient computation
                   ( 1/2 J_A     J_B   )
            C_up = (                   )
                   (    0      1/2 J_D )

        :return: result, MUp, C_up matrix
        """
        return MUp(0.5 * torch.ones((self.k, self.k)),
                   torch.ones((self.k, self.d - self.k)),
                   0.5 * torch.ones(self.d - self.k),
                   self.k,
                   device=self.device)

    def sample(self, mu: Union[int, float, np.array] = 0.0, n: int = 1) -> np.array:
        """
        Sample z ~ N(mu, Sigma) with covariance matrix
          Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2}):
        z = mu + U @ eps_rank + diag(0_k, m_d^{-1} @ eps_diag),
          where eps_rank ~ N(0, I_k), eps_diag ~ N(0, I_{d-k})
        Or:
        z = mu + B^{-T} @ eps
          where eps ~ N(0, I_d)

        :param mu: Union[int, float, np.array], mean value for Normal distribution
        :param n: int, number of samples to draw from
        :return: z, np.array of shape (n, d), array of n samples of N(mu, S^{-1})
        """
        # eps_rank = torch.randn((self.k, n), device=self.device)
        # eps_diag = torch.randn((self.d - self.k, n), device=self.device)
        # z = mu + (self.U() @ eps_rank + torch.cat((torch.zeros((self.k, n), device=self.device),
        #                                            self.m_d_inv() * eps_diag), 0)).T

        eps = torch.randn((self.d, n), device=self.device)
        z = mu + self.t().solve(eps)
        return z.T

    def _update(self, beta: float, eta: float, n: int, g: np.array, v: np.array, gamma: float = 1):
        """Perform update step
            B <- B h((1-beta) * C_up .* kappa_up(B^{-1} G_S B^{-T})),
        where h(M) := I + M + 1/2 M^2, kappa_up 'projects' to matrix group B_up
        by zeroing out entries.
        This function however avoids storing intermediate d x d matrices and
        computes the update step much more efficiently (see algorithm for details).

        :param beta: float, second order momentum strength
        :param eta: float, prior precision
        :param n: int, training set size
        :param g: np.array, gradient samples
        :param v: np.array, parameter samples (with mean zero, i.e. v ~ N(0, S^{-1})
        :param gamma: float, regularization parameter in ELBO
        :return: result, MUp, updated square root precision
        """
        assert(gamma >= 0)
        factor = gamma * eta / n
        # m_a_inv = self.m_a_inv()
        # m_d_inv = self.m_d_inv()

        identity = torch.eye(self.k, device=self.device)
        m_a_damp = self.m_a + self.damping * identity
        m_d_damp = self.m_d + self.damping

        m_a = torch.zeros_like(self.m_a, device=self.device)
        m_b = torch.zeros_like(self.m_b, device=self.device)
        m_d = torch.zeros_like(self.m_d, device=self.device)

        if self.k == 0:
            m_d += n * torch.mean(v[:, self.k:] * g[:, self.k:], axis=0)
        elif self.k == self.d:
            x_1 = self.m_a.T @ v[:, :self.k]
            y_1 = solve(m_a_damp, g[:, :self.k]).transpose(1, 2)

            M = torch.mean(x_1 @ y_1, axis=0)
            m_a += n/2 * (M + M.T)

            if gamma > 0:
                # gamma * eta / n * B_A^{-T} B_A^{-1} - gamma * I
                m_a += factor * solve(m_a_damp @ m_a_damp.T, identity) - gamma * identity
        else:
            x_1 = self.m_a.T @ v[:, :self.k]
            x_2 = self.m_b.T @ v[:, :self.k] + self.m_d * v[:, self.k:]
            y_1 = solve(m_a_damp.unsqueeze(0), g[:, :self.k]).transpose(1, 2)
            y_1 -= solve(m_a_damp.T.unsqueeze(0), self.m_b @ (g[:, self.k:] / m_d_damp)).transpose(1, 2)
            y_2 = g[:, self.k:] / m_d_damp

            M = torch.mean(x_1 @ y_1, axis=0)
            m_a += n/2 * (M + M.T)
            m_b += n/2 * torch.mean(x_1 @ y_2.transpose(1, 2), axis=0)
            m_b += n/2 * torch.mean(x_2 @ y_1, axis=0).T
            m_d += n * torch.mean(((self.m_b.T @ v[:, :self.k]) / m_d_damp + v[:, self.k:]) * g[:, self.k:], axis=0)

            if gamma > 0:
                identity = torch.eye(self.k, device=self.device)
                m_a += factor * solve(m_a_damp @ m_a_damp.T, identity) - gamma * identity
                m_b -= factor * solve(m_a_damp @ m_a_damp.T, self.m_b / m_d_damp.T)

        if gamma > 0:
            x = solve(m_a_damp, self.m_b)
            m_d += factor * (1 + torch.sum(x ** 2, axis=0)).reshape(-1, 1) / (m_d_damp ** 2) - gamma
        # print(f"update: {h((1-beta) * MUp(0.5 * m_a, m_b, 0.5 * m_d, self.k, device=self.device))}")

        # We avoid computing C_up * kappa_up(M) by simply multiplying the scalar 
        # values in the respective blocks
        # This returns B @ h(lr * C_up * kappa_up(B^{-1} G_S B^{-T}))
        return self @ h((1-beta) * MUp(0.5 * m_a, m_b, 0.5 * m_d, self.k, device=self.device, damping=self.damping))


class MLow:
    def __init__(self, m_a: np.array, m_c: np.array, m_d: np.array, k: int,
                 device: str = None, damping: float = 0.0) -> None:
        """Block lower triangular matrix class
                ( m_a    0  )
            M = (           )
                ( m_c   m_d )

        :param m_a: np.array of shape (k, k), first block
        :param m_c: np.array of shape (d-k, k), second block
        :param m_d: np.array of shape (d-k), diagonal third block
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.k = k
        self.d = m_d.shape[0] + k
        self.device = device
        self.m_a = m_a.to(device)
        self.m_c = m_c.to(device)
        self.m_d = m_d.to(device).reshape((-1, 1))
        self.shape = self.size()
        self.damping = damping
        self.inverse = None
        self.a_inv = None
        self.d_inv = None

    @staticmethod
    def eye(d: int, k: int, device: str = None, damping: float = 0.0):
        """Identity matrix of shape d x d represented as MLow.

        :param d: int, size of matrix
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        :return: result: MLow, identity matrix represented as MLow
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return MLow(torch.eye(k, k), torch.zeros(d-k, k), torch.ones(d-k), k, device=device, damping=damping)

    @staticmethod
    def zeros(d: int, k: int, device: str = None, damping: float = 0.0):
        """Zero matrix of shape d x d represented as MLow.

        :param d: int, size of matrix
        :param k: int, size of first block m_a
        :param device: str, torch device to run operations on (GPU or CPU)
        :param damping: float, damping term for inversion or linear system solution
        :return: result: MLow, zero matrix represented as MLow
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return MLow(torch.zeros(k, k), torch.zeros(d-k, k), torch.zeros(d-k), k, device=device, damping=damping)

    def size(self):
        """Return size as torch.Size object.

        :return: size, torch.Size, (d, d) tuple specifying size
        """
        return torch.Size((self.d, self.d))

    def to(self, device: str):
        """Move object onto device.

        :param device: str, torch device to run operations on (GPU or CPU)
        :return: self, MUp matrix on specified device
        """
        self.device = device
        self.m_a = self.m_a.to(device)
        self.m_c = self.m_c.to(device)
        self.m_d = self.m_d.reshape((-1, 1)).to(device)
        return self

    def t(self):
        """Return transpose of block matrix which results in block lower triangular matrix.

        :return: transpose, MUp, transpose of object
        """
        return MUp(self.m_a.T, self.m_c.T, self.m_d, self.k, device=self.device, damping=self.damping)

    def __repr__(self) -> str:
        """String representation of class including blocks: m_a, m_c, m_d

        :return: string, str, string representation of class
        """
        string = "m_a: \n\t"
        string += str(self.m_a)
        string += "\nm_c: \n\t"
        string += str(self.m_c)
        string += "\nm_d: \n\t"
        string += str(self.m_d)
        return string

    def __add__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Addition X + M

        :param x: Union[int, float, np.array, MLow], argument for elementwise addition; for scalar,
            broadcasted addition
        :return: result, MLow, result of addition
        """
        x = x.to(self.device)
        if isinstance(x, MLow):
            assert((self.size() == x.size()) and (self.k == x.k))
            return MLow(x.m_a + self.m_a,
                        x.m_c + self.m_c,
                        x.m_d + self.m_d,
                        self.k,
                        device=self.device,
                        damping=self.damping)
        elif isinstance(x, (int, float)):
            return MLow(x + self.m_a,
                        x + self.m_c,
                        x + self.m_d,
                        self.k,
                        device=self.device,
                        damping=self.damping)
        elif len(x.shape) == 2:
            assert(x.shape == self.shape)
            return MLow(x[:self.k, :self.k] + self.m_a,
                        x[self.k:, :self.k] + self.m_c,
                        torch.diag(x[self.k:, self.k:]).reshape(-1, 1) + self.m_d,
                        self.k,
                        device=self.device,
                        damping=self.damping)
        else:
            raise ValueError()

    def __radd__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Addition X + M

        :param x: Union[int, float, np.array, MLow], argument for elementwise addition; for scalar,
            broadcasted addition
        :return: result, MLow, result of addition
        """
        return self + x

    def __truediv__(self, x: Union[int, float]) -> np.array:
        """(Elementwise) Division M / x

        :param x: Union[int, float, np.array, MLow], argument for elementwise division
        :return: result, MLow, result of division
        """
        return 1/x * self

    def __neg__(self):
        """Unary minus.

        :return: result, MUp, negated entries
        """
        return MUp(-self.m_a, -self.m_c, -self.m_d, self.k, self.device, self.damping)

    def __mul__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Multiplication X * M

        :param x: Union[int, float, np.array, MLow], argument for elementwise multiplication; for scalar,
            broadcasted multiplication
        :return: result, MLow, result of multiplication
        """
        if isinstance(x, MUp):
            assert(self.shape == x.shape)
            assert(x.k == self.k)
            return MUp(x.m_a * self.m_a,
                       torch.zeros_like(x.m_b, device=self.device),
                       x.m_d * self.m_d,
                       self.k,
                       device=self.device,
                       damping=self.damping)
        elif isinstance(x, MLow):
            assert(self.shape == x.shape)
            assert(x.k == self.k)
            return MLow(x.m_a * self.m_a,
                        x.m_c * self.m_c,
                        x.m_d * self.m_d,
                        self.k,
                        device=self.device,
                        damping=self.damping)
        elif isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and (x.ndim == 0)):
            return MLow(x* self.m_a,
                        x * self.m_c,
                        x * self.m_d,
                        self.k,
                        device=self.device,
                        damping=self.damping)
        else:
            # Implement np.array and torch.tensor cases
            raise ValueError()

    def __rmul__(self, x: Union[int, float, np.array]) -> np.array:
        """(Elementwise) Multiplication X * M

        :param x: Union[int, float, np.array, MLow], argument for elementwise multiplication; for scalar,
            broadcasted multiplication
        :return: result, MLow, result of multiplication
        """
        return self * x

    def __matmul__(self, x: Union[int, float, np.array]) -> np.array:
        """Matrix multiplication M @ X

        :param x: Union[int, float, np.array], scalar, vector, or matrix for matrix multiplication
        :return: result, result of matrix multiplication with B
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
                        device=self.device,
                        damping=self.damping)
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

    def solve(self, b: np.array) -> np.array:
        """Solve (B + damping * I) x = b

        :param b: np.array of shape (d, 1), Right hand side of linear system of equations
        :return: result, np.array of shape (d, 1) as solution of dampened linear system
        """
        assert(b.shape[0] == self.d)
        result = torch.zeros_like(b, device=self.device)
        identity = torch.eye(self.k, device=self.device)
        m_a = self.m_a + self.damping * identity
        m_d = self.m_d + self.damping
        result[:self.k] = solve(m_a, b[:self.k])
        result[self.k:] = (-self.m_c @ result[:self.k] + b[self.k:]) / m_d
        return result

    def inv(self):
        """Calculate inverse of lower triangular block matrix
                ( m_a    0  )             ( m_a^{-1}                     0    )
            M = (           ) => M^{-1} = (                                   )
                ( m_c   m_d )             ( -m_d^{-1} m_c m_a^{-1}   m_d^{-1} )

        :return: self.inverse, MUp representation of inverse
        """
        if self.inverse is None:
            m_a_inv = self.m_a_inv()
            m_d_inv = self.m_d_inv()
            self.inverse = MLow(m_a_inv, -m_d_inv * (self.m_c @ m_a_inv), m_d_inv, self.k, device=self.device)
        return self.inverse

    def m_a_inv(self) -> np.array:
        """Calculate dampened inverse for invertible k x k matrix block m_a

        :return: self.a_inv, torch.Tensor, dampened inverse of first block
        """
        if self.a_inv is None:
            identity = torch.eye(self.k, device=self.device)
            self.a_inv = solve(self.m_a + self.damping * identity, identity)

        return self.a_inv

    def m_d_inv(self) -> np.array:
        """Compute dampened inverse of diagonal block.

        :return: self.d_inv, torch.Tensor, dampened inverse of diagonal block
        """
        # m_d is diagonal matrix
        if self.d_inv is None:
            self.d_inv = 1/(self.m_d + self.damping)
        return self.d_inv

    def add_id(self, alpha: float = 1):
        """Add alpha * I to M, i.e.,
                            ( m_a1 + alpha * I            0       )
            M + alpha * I = (                                     )
                            (       m_c          m_d1 + alpha * I )

        :param alpha: float, factor to identity matrix to be added to M
        :return: result, MLow, result of M + alpha * I
        """
        return MLow(
            self.m_a + alpha * torch.eye(self.k, device=self.device),
            self.m_c,
            self.m_d + alpha,
            self.k,
            self.device,
            self.damping
        )

    def sample(self, mu: Union[float, np.array] = 0, n: int = 1) -> np.array:
        """
        Sample z ~ N(mu, Sigma) with covariance matrix
          Sigma = S^{-1} = B^{-T} B^{-1} = U U^T + diag(0_k, m_d^{-2}):
        z = mu + U @ eps_rank + diag(0_k, m_d^{-1} @ eps_diag),
          where eps_rank ~ N(0, I_k), eps_diag ~ N(0, I_{d-k})
        """

        eps = torch.randn((self.d, n), device=self.device)
        z = mu + self.t().solve(eps)
        return z.T

    def _update(self, beta: float, eta: float, n: int, g: np.array, v: np.array, gamma: float = 1):
        """Perform update step
            B <- B h(lr * C_low .* kappa_low(B^{-1} G_S B^{-T})),
        where h(M) := I + M + 1/2 M^2, kappa_up 'projects' to matrix group B_low
        by zeroing out entries.
        This function however avoids storing intermediate d x d matrices and
        computes the update step much more efficiently (see algorithm for details).

        :param beta: float, second order momentum strength
        :param eta: float, prior precision
        :param n: int, training set size
        :param g: np.array, gradient samples
        :param v: np.array, parameter samples (with mean zero, i.e. v ~ N(0, S^{-1})
        :param gamma: float, regularization parameter in ELBO
        :return: result, MLow, updated square root precision
        """
        assert(gamma >= 0)
        factor = gamma * eta / n
        m_a_inv = self.m_a_inv()
        m_d_inv = self.m_d_inv()

        m_a = torch.zeros_like(self.m_a, device=self.device)
        m_c = torch.zeros_like(self.m_c, device=self.device)
        m_d = torch.zeros_like(self.m_d, device=self.device)

        # Edge case handling for k = 0 and k = d
        if self.k == 0:
            m_d += n * torch.mean((self.m_d ** 2) * v[:, self.k:] * g[:, self.k:], axis=0)
        elif self.k == self.d:
            x_1 = self.m_a.T @ v[:, :self.k]
            y_1 = g[:, :self.k].transpose(1, 2) @ self.m_a
            M = torch.mean(x_1 @ y_1, axis=0)
            m_a += n/2 * (M + M.T)
            if gamma > 0:
                identity = torch.eye(self.k, device=self.device)
                m_a += factor * m_a_inv.T @ (identity + self.m_c.T @ (m_d_inv ** 2 * self.m_c)) @ m_a_inv - gamma * identity
        else:
            x_1 = self.m_a.T @ v[:, :self.k] + self.m_c.T @ v[:, self.k:]
            x_2 = self.m_d * v[:, self.k:]
            y_1 = g[:, :self.k].transpose(1, 2) @ self.m_a + g[:, self.k:].transpose(1, 2) @ self.m_c
            y_2 = (g[:, self.k:] * self.m_d).transpose(1, 2)

            M = torch.mean(x_1 @ y_1, axis=0)
            m_a = n/2 * (M + M.T)
            m_c += n/2 * torch.mean(x_1 @ y_2, axis=0).T
            m_c += n/2 * torch.mean(x_2 @ y_1, axis=0)
            m_d += n * torch.mean((self.m_d ** 2) * v[:, self.k:] * g[:, self.k:], axis=0)

            if gamma > 0:
                identity = torch.eye(self.k, device=self.device)
                m_a += factor * m_a_inv.T @ (identity + self.m_c.T @ ((m_d_inv ** 2) * self.m_c)) @ m_a_inv - gamma * identity
                m_c += -factor * (m_d_inv ** 2) * self.m_c @ m_a_inv

        if gamma > 0:
            m_d += factor * (m_d_inv ** 2) - gamma
        # print(f"update: {h((1-beta) * MLow(0.5 * m_a, m_c, 0.5 * m_d, self.k, device=self.device))}")

        # We avoid computing C_up * kappa_up(B^{-1} G_S B^{-T}) by simply multiplying the scalar
        # values in the respective blocks
        # This returns B @ h((1-beta) * C_up * kappa_up(B^{-1} G_S B^{-T}))
        return self @ h((1-beta) * MLow(0.5 * m_a, m_c, 0.5 * m_d, self.k, device=self.device, damping=self.damping))


class RankMatrix:
    def __init__(self, x: np.array = 0, y: np.array = 0, device: str = None) -> None:
        '''
        Representation of x @ y^T
        '''
        assert(x.shape[1] == y.shape[1])
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.x = x
        self.y = y
        self.k = x.shape[1]
        self.device = device
        self.shape = (x.shape[0], y.shape[0])

    def full(self) -> np.array:
        return self.x @ self.y.T

    def t(self):
        '''
        (x @ y^T)^T = y @ x^T = RankMatrix(y, x)
        '''
        return RankMatrix(self.y, self.x, device=self.device)

    def __matmul__(self, other: np.array) -> np.array:
        '''
        self @ other = x @ y.T @ other
        '''
        if isinstance(other, (MUp, MLow)):
            return RankMatrix(self.x, other.t() @ self.y, device=self.device)
        elif isinstance(other, RankMatrix):
            # self @ other = x_1 @ y_1^T @ x_2 @ y_2^T = dot(y_1, x_2) * x_1 @ y_2^T
            return torch.dot(self.y, other.x) * RankMatrix(self.x, other.y, device=self.device)
        elif (len(other.shape) == 1) or (other.shape[1] == 1):
            # other a vector
            # x @ y^T @ other = dot(y, other) * x
            return torch.dot(self.y, other) * self.x
        else:
            # other a matrix
            # x @ y^T @ other = x @ (other^T @ y)^T
            return RankMatrix(self.x, other.T @ self.y, device=self.device)

    def __rmatmul__(self, other: np.array) -> np.array:
        '''
        other @ self = other @ x @ y.T
        '''
        if isinstance(other, (MUp, MLow)):
            return RankMatrix(other @ self.x, self.y, device=self.device)
        elif isinstance(other, RankMatrix):
            # other @ self = x_2 @ y_2^T @ x_1 @ y_1^T = dot(y_2, x_1) * x_2 @ y_1^T
            return torch.dot(other.y, self.x) * RankMatrix(other.x, self.y, device=self.device)
        elif (len(other.shape) == 1) or (other.shape[1] == 1):
            # other a vector
            return (other @ self.x) @ self.y.T
        else:
            # other a matrix
            # x @ y^T @ other = x @ (other^T @ y)^T
            return RankMatrix(other @ self.x, self.y, device=self.device)


class BlockTriangular:
    def __init__(self, diag_blocks: List[Union[MUp, MLow]], bandwidth: int = 0,
                 off_diag_blocks: List[List[RankMatrix]] = None) -> None:
        self.diag_blocks = diag_blocks
        self.off_diag_blocks = off_diag_blocks
        self.bandwidth = bandwidth # width of off-diagonal block
        self.block_sizes = []
        for diag in self.diag_blocks:
            self.block_sizes.append(diag.k)
        assert(((bandwidth == 0) and (off_diag_blocks is None)) or (bandwidth == len(off_diag_blocks)))

    def __matmul__(self, other: np.array) -> np.array:
        '''
        self @ other
        '''
        # Chunk other into blocks
        other_blocked = []
        for k in self.block_sizes:
            other_blocked.append(other[:k])
            other = other[k:]

        # Perform matrix multiplication blockwise first over diagonals, then off-diagonals
        result = 0
        for i, diag in enumerate(self.diag_blocks):
            result += diag @ other_blocked[i]
        for band in range(self.bandwidth):
            for i, off_diag in enumerate(self.off_diag_blocks[band]):
                result += off_diag @ other_blocked[band + 1 + i]
        return result

    def solve(self, other: np.array) -> np.array:
        # Chunk other into blocks
        other_blocked = []
        solution_blocked = []
        for k in self.block_sizes:
            other_blocked.append(other[:k])
            other = other[k:]

        # Backwards substitution, Thomas algorithm (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)
        n = len(self.diag_blocks)
        solution_blocked.insert(0, self.diag_blocks[n].solve(other_blocked[n]))
        for i, diag in reversed(enumerate(self.diag_blocks[:-1])):
            solution_blocked.insert(0, diag.solve(other_blocked[i] - self.off_diag_blocks[i] @ solution_blocked[i+1]))
        solution = torch.stack(solution_blocked, dim=0)
        return solution

    def det(self) -> float:
        return np.prod(list(map(lambda x: x.det(), self.diag_blocks)))

    def trace(self) -> float:
        return np.sum(list(map(lambda x: x.trace(), self.diag_blocks)))


def h(x: Union[MUp, MLow]) -> Union[MUp, MLow]:
    """Return quadratic approximation to exponential function
        I + X + 1/2 * X @ X

    :param x: Union[MUp, MLow], argument
    :return: result, Union[MUp, MLow], result of quadratic approximation
    """
    return x.add_id() + 0.5 * x @ x
