import torch
import numpy as np
import torch.nn as nn
from typing import Union, Tuple, List, Callable

from torch.utils.data import DataLoader
from optimizers.noisy_optimizer import *
from metrics import *


class HyperDeepEnsemble():
    def __init__(self, models, optimizers, num_classes=10, input_shape=(3, 32, 32), device: str = None, *args, **kwargs):
        self.__name__ = 'HyperDeepEnsemble'
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.num_classes = num_classes
        self.models = models
        self.optimizers = optimizers
        self.num_params = np.sum([model.num_params for model in self.models])
        self.__name__ = self.models[0].__name__

    def __call__(self, x, mc_samples=1):
        logits = torch.zeros((len(self.models), mc_samples, x.shape[0], self.num_classes), device=self.device)
        for i, model, optimizer in enumerate(zip(self.models, self.optimizers)):
            with Sampler(optimizer):
                for j in range(mc_samples):
                    optimizer.sample_weight()
                    logits[i][j] = model(x)
        return logits

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, metrics: List[Callable] = [],
                 loss_fn: Callable = nn.CrossEntropyLoss(), n_bins=10, **kwargs) -> Tuple[float, dict, dict]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals, bin_data), Tuple[float, dict, dict], tuple of loss and specified metrics on validation set
        """
        # Set model to evaluation mode!
        for model in self.models:
            model.eval()

        loss = 0.0
        metric_vals = {}
        for metric in metrics:
            metric_vals[metric.__name__] = 0.0
        bin_accuracies = np.zeros(n_bins, dtype=float)
        bin_confidences = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        for i, data in enumerate(data_loader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self(images)
            # Take logits w.r.t. variational posterior, then across the models
            loss += loss_fn(logits.mean(1).mean(0), labels).item()
            for metric in metrics:
                metric_vals[metric.__name__] += metric(logits.mean(1).mean(0), labels).item()

            # Take preds w.r.t. variational posterior, then across the models
            confidences, preds = logits.softmax(-1).mean(1).softmax(-1).mean(0).max(-1)

            labels = labels.detach().cpu().numpy()
            confidences = confidences.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            indices = np.digitize(confidences, bins, right=True)

            for b in range(n_bins):
                selected = np.where(indices == b + 1)[0]
                if len(selected) > 0:
                    bin_accuracies[b] += np.sum(labels[selected] == preds[selected])
                    bin_confidences[b] += np.sum(confidences[selected])
                    bin_counts[b] += len(selected)

            # Write to TensorBoard
            # writer.add_scalar("Loss", loss, counter)

        for metric in metrics:
            metric_vals[metric.__name__] /= len(data_loader)
        loss /= len(data_loader)
        print(loss, metric_vals)

        # Divide each bin by its bin count and avoid division by zero!
        bin_accuracies /= np.where(bin_counts > 0, bin_counts, 1)
        bin_confidences /= np.where(bin_counts > 0, bin_counts, 1)
        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        mce = np.max(gaps)
        bin_data = dict(
            accuracies=bin_accuracies,
            confidences=bin_confidences,
            counts=bin_counts,
            bins=bins,
            avg_accuracy=avg_acc,
            avg_confidence=avg_conf,
            expected_calibration_error=ece,
            max_calibration_error=mce
        )
        return loss, metric_vals, bin_data


class Sampler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __enter__(self):
        self.optimizer._stash_param_averages()

    def __exit__(self, *args):
        self.optimizer._restore_param_averages()