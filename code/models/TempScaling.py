# Reference: https://github.com/gpleiss/temperature_scaling
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from typing import List, Callable, Tuple

from metrics import *
from util import *


class TempScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(TempScaling, self).__init__()
        self.device = model.device
        self.model = model
        self.num_classes = model.num_classes
        self.temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        self.__name__ = model.__name__
        self.num_params = model.num_params + 1
        self.summary = model.summary

    def forward(self, images):
        logits = self.model(images)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(labels)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    @torch.no_grad()
    def evaluate(
            self, data_loader: DataLoader, metrics: List[Callable] = [],
            loss_fn: Callable = nn.CrossEntropyLoss(), n_bins=10, **kwargs
    ) -> Tuple[float, dict, dict, np.array]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals, bin_data), Tuple[float, dict, dict], tuple of loss and specified metrics on validation set
        """
        # Set model to evaluation mode!
        self.model.eval()

        logits = []
        labels_list = []
        for i, data in enumerate(data_loader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits.append(self(images))
            labels_list.append(labels)
        logits = torch.cat(logits)
        labels = torch.cat(labels_list)

        loss = loss_fn(logits, labels).item()
        metric_vals = {}
        for metric in metrics:
            metric_vals[metric.__name__] = metric(logits, labels).item()

        bin_data = get_bin_data(logits, labels, num_classes=self.num_classes, n_bins=n_bins)
        uncertainty = get_uncertainty(logits)

        print(f"\tNLL = {loss}\n")
        print("\t{:<25} {:<10}".format('Metric', 'Value'))
        for k, v in metric_vals.items():
            print("\t{:<25} {:<10.3f}".format(k, v))
        return loss, metric_vals, bin_data, uncertainty


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


def get_uncertainty(logits):
    return dict(
        model_uncertainty=model_uncertainty(logits).detach().cpu().numpy(),
        predictive_uncertainty=predictive_uncertainty(logits).detach().cpu().numpy(),
        # data_uncertainty=data_uncertainty(logits).detach().cpu().numpy()
    )


def get_bin_data(logits, labels, num_classes=-1, n_bins=10):
    if num_classes == -1:
        num_classes = F.one_hot(labels).shape[-1]
    probs = logits.softmax(-1)
    if len(probs.shape) == 3:
        probs = probs.mean(0)
    if len(probs.shape) == 4:
        probs = probs.mean(1).mean(0)
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