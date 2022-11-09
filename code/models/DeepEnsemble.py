import torch
import numpy as np
import torch.nn as nn
from typing import Union, Tuple, List, Callable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optimizers.noisy_optimizer import *
from metrics import *
from util import *


class DeepEnsemble:
    def __init__(self, models, num_classes=10, device: str = None, *args, **kwargs):
        self.__name__ = 'DeepEnsemble'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_classes = num_classes
        self.models = models
        self.num_params = np.sum([model.num_params for model in self.models])
        self.__name__ = self.models[0].__name__
        self.summary = None
        print(f"Using DeepEnsemble with {len(self.models)} models.")

    def __call__(self, x):
        logits = torch.zeros((len(self.models), x.shape[0], self.num_classes), device=self.device)
        for i, model in enumerate(self.models):
            logits[i] = model(x)
        return logits

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, metrics: List[Callable] = [],
                 loss_fn: Callable = nn.NLLLoss(), n_bins=10, **kwargs) -> Tuple[float, dict, dict, np.array]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals, bin_data), Tuple[float, dict, dict], tuple of loss and specified metrics on validation set
        """
        # Set model to evaluation mode!
        for model in self.models:
            model.eval()

        logits = []
        labels_list = []
        for i, model in enumerate(self.models):
            model_logits = []
            for j, data in enumerate(data_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                model_logits.append(model(images))
                if i == 0:
                    labels_list.append(labels)
            model_logits = torch.cat(model_logits)
            logits.append(model_logits)
        logits = torch.stack(logits)
        labels = torch.cat(labels_list)
        loss = loss_fn(F.log_softmax(logits, dim=-1).mean(0), labels).item()
        metric_vals = {}
        for metric in metrics:
            metric_vals[metric.__name__] = metric(logits, labels).item()

        bin_data = get_bin_data(logits, labels, num_classes=self.num_classes, n_bins=n_bins)
        uncertainty = get_uncertainty(logits)
        print(loss, metric_vals)
        return loss, metric_vals, bin_data, uncertainty


def get_uncertainty(logits):
    return dict(
        model_uncertainty=model_uncertainty(logits),
        predictive_uncertainty=predictive_uncertainty(logits),
        # data_uncertainty=data_uncertainty(logits)
    )


def get_bin_data(logits, labels, num_classes=-1, n_bins=10):
    if num_classes == -1:
        num_classes = F.one_hot(labels).shape[-1]
    probs = logits.softmax(-1)
    if len(probs.shape) == 3:
        probs = probs.mean(0)
    num_classes = torch.tensor(num_classes, dtype=float)
    uncertainties = 1/torch.log(num_classes) * predictive_uncertainty(logits)
    confidences, preds = probs.max(-1)

    uncertainties = uncertainties.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    bin_accuracies = np.zeros(n_bins, dtype=float)
    bin_confidences = np.zeros(n_bins, dtype=float)
    r_bin_counts = np.zeros(n_bins, dtype=int)
    bin_errors = np.zeros(n_bins, dtype=float)
    bin_uncertainties = np.zeros(n_bins, dtype=float)
    u_bin_counts = np.zeros(n_bins, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    r_indices = np.digitize(confidences, bins, right=True)
    u_indices = np.digitize(uncertainties, bins, right=True)

    for b in range(n_bins):
        selected = np.where(r_indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(labels[selected] == preds[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            r_bin_counts[b] = len(selected)
        selected = np.where(u_indices == b + 1)[0]
        if len(selected) > 0:
            bin_errors[b] = np.mean(labels[selected] != preds[selected])
            bin_uncertainties[b] = np.mean(uncertainties[selected])
            u_bin_counts[b] = len(selected)

    # Divide each bin by its bin count and avoid division by zero!
    bin_accuracies /= np.where(r_bin_counts > 0, r_bin_counts, 1)
    bin_confidences /= np.where(r_bin_counts > 0, r_bin_counts, 1)
    avg_acc = np.sum(bin_accuracies * r_bin_counts) / np.sum(r_bin_counts)
    avg_conf = np.sum(bin_confidences * r_bin_counts) / np.sum(r_bin_counts)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * r_bin_counts) / np.sum(r_bin_counts)
    mce = np.max(gaps)

    bin_errors /= np.where(u_bin_counts > 0, u_bin_counts, 1)
    bin_uncertainties /= np.where(u_bin_counts > 0, u_bin_counts, 1)
    avg_err = np.sum(bin_errors * u_bin_counts) / np.sum(u_bin_counts)
    avg_uncert = np.sum(bin_uncertainties * u_bin_counts) / np.sum(u_bin_counts)
    gaps = np.abs(bin_errors - bin_uncertainties)
    uce = np.sum(gaps * u_bin_counts) / np.sum(u_bin_counts)
    muce = np.max(gaps)

    bin_data = dict(
        accuracies=bin_accuracies,
        confidences=bin_confidences,
        errors=bin_errors,
        uncertainties=bin_uncertainties,
        r_counts=r_bin_counts,
        u_counts=u_bin_counts,
        bins=bins,
        avg_accuracy=avg_acc,
        avg_confidence=avg_conf,
        avg_error=avg_err,
        avg_uncertainty=avg_uncert,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        expected_uncertainty_error=uce,
        max_uncertainty_error=muce
    )
    return bin_data


def entropy(probs, labels=None):
    return -torch.sum(probs * torch.log(probs + torch.finfo().tiny), axis=-1)


def model_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    pred_uncert = predictive_uncertainty(logits)
    data_uncert = data_uncertainty(logits)
    return pred_uncert - data_uncert


def predictive_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(-1)
    if len(probs.shape) == 4:
        probs = probs.mean(1)
    return entropy(probs.mean(0))


def data_uncertainty(logits, labels=None):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    probs = logits.softmax(-1)
    if len(probs.shape) == 4:
        probs = probs.mean(1)
    return entropy(probs).mean(0)