from torchmetrics.functional import accuracy, precision, recall, f1_score, calibration_error
import torch
import torch.nn.functional as F


class ECE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'ece'

    def __call__(self, logits, labels):
        return calibration_error(logits, labels, n_bins=self.n_bins, norm='l1')


class MCE:
    def __init__(self, n_bins=10):
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.__name__ = 'mce'

    def __call__(self, logits, labels):
        return calibration_error(logits, labels, n_bins=self.n_bins, norm='max')


class UCE:
    def __init__(self, num_classes, n_bins=10):
        assert(num_classes > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.num_classes = torch.tensor(num_classes, dtype=float)
        self.__name__ = 'uce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        preds = probs.argmax(-1)
        uncertainties = -1/torch.log(self.num_classes) * torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_counts = torch.zeros(self.n_bins, dtype=int, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors[b] = torch.mean((labels[selected] != preds[selected]).float())
                bin_uncertainties[b] = torch.mean(uncertainties[selected])
                bin_counts[b] = len(selected)

        gaps = torch.abs(bin_errors - bin_uncertainties)
        uce = torch.sum(gaps * bin_counts) / torch.sum(bin_counts)
        return uce


class MUCE:
    def __init__(self, num_classes, n_bins=10):
        assert(num_classes > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.num_classes = torch.tensor(num_classes, dtype=float)
        self.__name__ = 'muce'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        preds = probs.argmax(-1)
        uncertainties = -1/torch.log(self.num_classes) * torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors[b] = torch.mean((labels[selected] != preds[selected]).float())
                bin_uncertainties[b] = torch.mean(uncertainties[selected])

        gaps = torch.abs(bin_errors - bin_uncertainties)
        muce = torch.max(gaps)
        return muce


class TopkAccuracy:
    def __init__(self, top_k=5):
        self.top_k = top_k
        if top_k == 1:
            self.__name__ = 'accuracy'
        else:
            self.__name__ = f"top_{top_k}_accuracy"

    def __call__(self, logits, labels):
        return accuracy(logits, labels, top_k=self.top_k)


class TopkECE:
    def __init__(self, top_k=5, n_bins=10):
        assert(top_k > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.top_k = top_k
        if top_k == 1:
            self.__name__ = 'ece'
        else:
            self.__name__ = f"top_{top_k}_ece"

    def __call__(self, logits, labels):
        confidences, _ = logits.softmax(-1).max(-1)
        _, preds_topk = logits.softmax(-1).topk(self.top_k)
        labels_topk = labels.unsqueeze(-1).expand_as(preds_topk)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(confidences, bins, right=True)

        bin_accuracies_topk = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_confidences = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_counts = torch.zeros(self.n_bins, dtype=int, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies_topk[b] = (labels_topk[selected] == preds_topk[selected]).sum(-1).float().mean()
                bin_confidences[b] = torch.mean(confidences[selected])
                bin_counts[b] = len(selected)

        gaps = torch.abs(bin_accuracies_topk - bin_confidences)
        ece_topk = torch.sum(gaps * bin_counts) / torch.sum(bin_counts)
        return ece_topk


class TopkUCE:
    def __init__(self, num_classes, top_k=5, n_bins=10):
        assert(top_k > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.num_classes = torch.tensor(num_classes, dtype=float)
        if top_k == 1:
            self.__name__ = 'uce'
        else:
            self.__name__ = f"top_{top_k}_uce"

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        _, preds_topk = logits.softmax(-1).topk(self.top_k)
        labels_topk = labels.unsqueeze(-1).expand_as(preds_topk)
        uncertainties = -1/torch.log(self.num_classes) * torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors_topk = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_counts = torch.zeros(self.n_bins, dtype=int, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors_topk[b] = (1 - (labels_topk[selected] == preds_topk[selected]).sum(-1)).float().mean()
                bin_uncertainties[b] = torch.mean(uncertainties[selected])
                bin_counts[b] = len(selected)

        gaps = torch.abs(bin_errors_topk - bin_uncertainties)
        uce_topk = torch.sum(gaps * bin_counts) / torch.sum(bin_counts)
        return uce_topk


class TopkMCE:
    def __init__(self, top_k=5, n_bins=10):
        assert(top_k > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.top_k = top_k
        if top_k == 1:
            self.__name__ = 'mce'
        else:
            self.__name__ = f"top_{top_k}_mce"

    def __call__(self, logits, labels):
        confidences, _ = logits.softmax(-1).max(-1)
        _, preds_topk = logits.softmax(-1).topk(self.top_k)
        labels_topk = labels.unsqueeze(-1).expand_as(preds_topk)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(confidences, bins, right=True)

        bin_accuracies_topk = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_confidences = torch.zeros(self.n_bins, dtype=float, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies_topk[b] = (labels_topk[selected] == preds_topk[selected]).sum(-1).mean()
                bin_confidences[b] = torch.mean(confidences[selected])

        gaps = torch.abs(bin_accuracies_topk - bin_confidences)
        mce_topk = torch.max(gaps)
        return mce_topk


class TopkMUCE:
    def __init__(self, num_classes, top_k=5, n_bins=10):
        assert(top_k > 0)
        assert(n_bins > 0)
        self.n_bins = n_bins
        self.top_k = top_k
        self.num_classes = torch.tensor(num_classes, dtype=float)
        if top_k == 1:
            self.__name__ = 'muce'
        else:
            self.__name__ = f"top_{top_k}_muce"

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        _, preds_topk = logits.softmax(-1).topk(self.top_k)
        labels_topk = labels.unsqueeze(-1).expand_as(preds_topk)
        uncertainties = -1/torch.log(self.num_classes) * torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)

        bins = torch.linspace(0.0, 1.0, self.n_bins + 1, device=logits.device)
        indices = torch.bucketize(uncertainties, bins, right=True)

        bin_errors_topk = torch.zeros(self.n_bins, dtype=float, device=logits.device)
        bin_uncertainties = torch.zeros(self.n_bins, dtype=float, device=logits.device)

        for b in range(self.n_bins):
            selected = torch.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_errors_topk[b] = (1 - (labels_topk[selected] == preds_topk[selected]).sum(-1)).float().mean(0)
                bin_uncertainties[b] = torch.mean(uncertainties[selected])

        gaps = torch.abs(bin_errors_topk - bin_uncertainties)
        mce_topk = torch.max(gaps)
        return mce_topk


class Brier:
    def __init__(self, num_classes=-1):
        self.num_classes = num_classes
        self.__name__ = 'brier'

    def __call__(self, logits, labels):
        probs = logits.softmax(-1)
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes)
        return ((probs - labels_onehot) ** 2).sum(-1).mean()


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
