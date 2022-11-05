from torchmetrics.functional import accuracy, precision, recall, f1_score, calibration_error
import torch
from bayesian_torch.utils.util import predictive_entropy, mutual_information


class ECE:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.__name__ = 'ece'

    def __call__(self, logits, labels):
        return calibration_error(logits, labels, n_bins=self.n_bins, norm='l1')


class MCE:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.__name__ = 'mce'

    def __call__(self, logits, labels):
        return calibration_error(logits, labels, n_bins=self.n_bins, norm='max')


class TopkAccuracy:
    def __init__(self, top_k=5):
        self.top_k = top_k
        if top_k == 1:
            self.__name__ = 'accuracy'
        else:
            self.__name__ = f"top_{top_k}_accuracy"

    def __call__(self, logits, labels):
        return accuracy(logits, labels, top_k=self.top_k)


class Rosenbrock:
    def __init__(self):
        pass

    def __call__(self, preds, labels=None):
        d = preds[0].nelement()
        # minimum in x^* = (1, ..., 1); f(x^*) = 0
        # 1/d * sum_{i=1}^{d-1} [100(w_{i+1) - w_i)^2 + (w_i - 1)^2]
        # x_min = torch.ones(d)
        return torch.mean(1/d * (torch.sum(100 * (preds[:, 1:] - preds[:, :-1]) + (preds[:, :-1] - 1) ** 2)))


class DixonPrice:
    def __init__(self):
        pass

    def __call__(self, preds, labels=None):
        d = preds[0].nelement()
        i = torch.arange(2, d + 1, dtype=float)

        # minimum in x_i^* = (2^{-(2^i -2) / 2^i}) for i = 1, ... d; f(x^*) = 0
        # 1/d * sum_{i=1}^{d-1} [100(w_{i+1) - w_i)^2 + (w_i - 1)^2]
        # x_min = 2 ** (-1 + 2 ** (1-i))
        return torch.mean(1/d * ((preds[:, 0] - 1) ** 2 +
                                 torch.sum(i * (2 * preds[:, 1:] ** 2 - preds[:, :-1]) ** 2)))
