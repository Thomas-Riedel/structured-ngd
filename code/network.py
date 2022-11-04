import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from optimizers.noisy_optimizer import *
from torchsummary import summary
from metrics import *
from models import *

from typing import Union, Tuple, List, Callable
import time


class Model(nn.Module):
    def __init__(self, model_type: str = 'ResNet20',
                 num_classes: int = 10, input_shape=(3, 32, 32),
                 device: str = None) -> None:
        """torch.nn.Module class

        :param model_type: str, specify what ResNet model to use
        :param num_classes: int, number of classes in dataset for classification
        :param device: str, torch device to run operations on (GPU or CPU)
        """
        super(Model, self).__init__()
        model_types_cifar = ['resnet' + str(n) for n in [20, 32, 44, 56, 110, 1202]]
        if model_type.lower() in model_types_cifar:
            print(model_type)
            print(f"Using {model_type} designed for CIFAR-10/100.")
            self.model = eval(f"{model_type}(num_classes=num_classes).to(device)")
        else:
            print(f"Using {model_type}.")
            self.model = eval(f"{model_type}(num_classes=num_classes).to(device)")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_weights()
        self.num_classes = num_classes
        self.device = device

        self.__name__ = model_type
        self.summary = summary(self.model, input_shape)
        self.num_params = np.sum(p.numel() for p in self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through model.

        :param x: torch.tensor, argument for forward pass
        :return: output, Tensor, model output
        """
        return self.model(x)

    def train(self, data_loader: DataLoader, optimizer: Union[Adam, NoisyOptimizer],
              epoch: int = 0, metrics: List[Callable] = [accuracy], eval_every: int = 100,
              loss_fn: Callable = nn.CrossEntropyLoss()) -> Tuple[List[float], dict, List[float]]:
        """Training loop for one epoch.

        :param data_loader: torch.utils.data.DataLoader, training dataset
        :param optimizer: Union[Adam, NoisyOptimizer], which optimizer to use
        :param epoch: int, current epoch (training will be done for only one single epoch!)
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param eval_every: int, running losses and metrics will be averaged and displayed after a certain number of
            iterations
        :param loss_fn: Callable, loss function to optimize
        :return: (iter_loss, iter_metrics, iter_time), Tuple[List[float], dict, List[float]], list of iterationwise
            losses, metrics and computation times for update
        """
        # Set model to training mode!
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {}
        epoch_time = 0.0

        running_loss = 0.0
        running_metrics = {}
        for metric in metrics:
            epoch_metrics[metric.__name__] = 0.0
            running_metrics[metric.__name__] = 0.0

        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            def closure():
                optimizer.zero_grad()
                preds = self(images)
                loss = loss_fn(preds, labels)
                loss.backward()
                return loss, preds

            # Perform forward pass, compute loss, backpropagate, update parameters
            with Timer(self.device) as t:
                loss, preds = optimizer.step(closure)
            epoch_time += t.elapsed_time
            # print(loss.item())
            # print(t.elapsed_time)

            # Record losses and metrics
            running_loss += loss.item()
            epoch_loss += loss.item()
            for metric in metrics:
                running_metrics[metric.__name__] += metric(preds, labels).item()
                epoch_metrics[metric.__name__] += metric(preds, labels).item()

            if i % eval_every == (eval_every - 1):
                print("===========================================")
                print(f"[{epoch + 1}, {i + 1}] Total loss: {running_loss / eval_every:.3f}")
                running_loss = 0.0
                for metric in metrics:
                    name = metric.__name__
                    print(f"\t{name}: {running_metrics[name] / eval_every:.3f}")
                    running_metrics[name] = 0.0
                print("===========================================")

        for metric in metrics:
            epoch_metrics[metric.__name__] /= len(data_loader)
        epoch_loss /= len(data_loader)
        return epoch_loss, epoch_metrics, epoch_time

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, metrics: List[Callable] = [],
                 loss_fn: Callable = nn.CrossEntropyLoss(),
                 optimizer = None, mc_samples: int = 1, n_bins=10) -> Tuple[float, dict, dict]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals, bin_data), Tuple[float, dict, dict], tuple of loss and specified metrics on validation set
        """
        if data_loader is None:
            return None, None, None
        if not isinstance(optimizer, NoisyOptimizer):
            mc_samples = 1
        assert(mc_samples >= 1)

        # Set model to evaluation mode!
        self.model.eval()

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

            with Sampler(optimizer):
                logits = torch.zeros((images.shape[0], self.num_classes), device=self.device)
                for i in range(mc_samples):
                    if isinstance(optimizer, NoisyOptimizer):
                        optimizer._sample_weight()
                    logits += self(images)
            logits /= mc_samples
            loss += loss_fn(logits, labels).item()
            for metric in metrics:
                metric_vals[metric.__name__] += metric(logits, labels).item()

            confidences, preds = logits.softmax(1).max(1)
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

    def collect(self, data_loader):
        for i, data in enumerate(data_loader, 0):
            image, label = data
            image = image.to(self.device)
            logit = self(image)
            pred = torch.argmax(logit, axis=1)
            if i == 0:
                labels, preds, logits = label, pred, logit
            else:
                labels = torch.cat((labels, label), dim=0)
                preds = torch.cat((preds, pred), dim=0)
                logits = torch.cat((logits, logit), dim=0)

        labels = labels.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        return labels, preds, logits

    @torch.no_grad()
    def compute_calibration(self, data_loader, n_bins: int = 10, optimizer = None, mc_samples: int = 1):
        """Collects predictions into bins used to draw a reliability diagram.
            Adapted code snippet by https://github.com/hollance/reliability-diagrams

        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            n_bins: number of bins

        The true_labels, pred_labels, confidences arguments must be NumPy arrays;
        pred_labels and true_labels may contain numeric or string labels.

        For a multi-class model, the predicted label and confidence should be those
        of the highest scoring class.

        Returns a dictionary containing the following NumPy arrays:
            accuracies: the average accuracy for each bin
            confidences: the average confidence for each bin
            counts: the number of examples in each bin
            bins: the confidence thresholds for each bin
            avg_accuracy: the accuracy over the entire test set
            avg_confidence: the average confidence over the entire test set
            expected_calibration_error: a weighted average of all calibration gaps
            max_calibration_error: the largest calibration gap across all bins
        """
        # assert(len(confidences) == len(pred_labels))
        # assert(len(confidences) == len(true_labels))
        assert(n_bins > 0)

        if not isinstance(optimizer, NoisyOptimizer):
            mc_samples = 1
        assert(mc_samples >= 1)

        # Set model to evaluation mode!
        self.model.eval()

        bin_accuracies = np.zeros(n_bins, dtype=float)
        bin_confidences = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        for data in data_loader:
            images, true_labels = data
            images = images.to(self.device)
            true_labels = true_labels.to(self.device)

            outputs = torch.zeros((images.shape[0], self.num_classes), device=self.device)
            with Sampler(optimizer):
                for i in range(mc_samples):
                    if isinstance(optimizer, NoisyOptimizer):
                        optimizer._sample_weight()
                    outputs += self(images)
                outputs /= mc_samples
            confidences, pred_labels = outputs.softmax(1).max(1)

            true_labels = true_labels.detach().cpu().numpy()
            confidences = confidences.detach().cpu().numpy()
            pred_labels = pred_labels.detach().cpu().numpy()

            indices = np.digitize(confidences, bins, right=True)

            for b in range(n_bins):
                selected = np.where(indices == b + 1)[0]
                if len(selected) > 0:
                    bin_accuracies[b] += np.sum(true_labels[selected] == pred_labels[selected])
                    bin_confidences[b] += np.sum(confidences[selected])
                    bin_counts[b] += len(selected)

        # Divide each bin by its bin count and avoid division by zero!
        bin_accuracies /= np.where(bin_counts > 0, bin_counts, 1)
        bin_confidences /= np.where(bin_counts > 0, bin_counts, 1)
        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        mce = np.max(gaps)

        return {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce
        }

    def init_weights(self, seed: Union[int, None] = None) -> None:
        """Initialize weights.

        :param seed: int, seed for random initialization
        """
        if not seed is None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_(0, 1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


class DeepEnsemble():
    def __init__(self, runs, input_shape=(3, 32, 32), device: str = None):
        datasets = list(set([run['dataset'] for run in runs]))
        assert(len(datasets) == 1)
        dataset = datasets[0]
        if dataset.lower() in ['stl10', 'cifar10']:
            num_classes = 10
        elif dataset.lower() == 'cifar100':
            num_classes = 100
        else:
            raise NotImplementedError(f"Dataset {dataset} not recognized!")
        self.__name__ = 'DeepEnsemble'

        self.models = []
        for run in runs:
            state_dict = torch.load(f"models/{run['timestamp']}.pt")
            model = Model(run['model_name'], num_classes=num_classes, input_shape=input_shape, device=device)
            model.load_state_dict(state_dict=state_dict)
            self.models.append(model)

    def __call__(self, x):
        logits = 0
        for model in self.models:
            logits += model(x)
        logits /= len(self.models)
        return logits

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, metrics: List[Callable] = [],
                 loss_fn: Callable = nn.CrossEntropyLoss(), n_bins=10) -> Tuple[float, dict, dict]:
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
            loss += loss_fn(logits, labels).item()
            for metric in metrics:
                metric_vals[metric.__name__] += metric(logits, labels).item()

            confidences, preds = logits.softmax(1).max(1)
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

    @torch.no_grad()
    def compute_calibration(self, data_loader, n_bins: int = 10, optimizer = None, mc_samples: int = 1):
        """Collects predictions into bins used to draw a reliability diagram.
            Adapted code snippet by https://github.com/hollance/reliability-diagrams

        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            n_bins: number of bins

        The true_labels, pred_labels, confidences arguments must be NumPy arrays;
        pred_labels and true_labels may contain numeric or string labels.

        For a multi-class model, the predicted label and confidence should be those
        of the highest scoring class.

        Returns a dictionary containing the following NumPy arrays:
            accuracies: the average accuracy for each bin
            confidences: the average confidence for each bin
            counts: the number of examples in each bin
            bins: the confidence thresholds for each bin
            avg_accuracy: the accuracy over the entire test set
            avg_confidence: the average confidence over the entire test set
            expected_calibration_error: a weighted average of all calibration gaps
            max_calibration_error: the largest calibration gap across all bins
        """
        # assert(len(confidences) == len(pred_labels))
        # assert(len(confidences) == len(true_labels))
        assert(n_bins > 0)

        if not isinstance(optimizer, NoisyOptimizer):
            mc_samples = 1
        assert(mc_samples >= 1)

        # Set models to evaluation mode!
        for model in self.models:
            model.eval()

        bin_accuracies = np.zeros(n_bins, dtype=float)
        bin_confidences = np.zeros(n_bins, dtype=float)
        bin_counts = np.zeros(n_bins, dtype=int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)

        for data in data_loader:
            images, true_labels = data
            images = images.to(self.device)
            true_labels = true_labels.to(self.device)

            outputs = self(images)
            confidences, pred_labels = outputs.softmax(1).max(1)

            true_labels = true_labels.detach().cpu().numpy()
            confidences = confidences.detach().cpu().numpy()
            pred_labels = pred_labels.detach().cpu().numpy()

            indices = np.digitize(confidences, bins, right=True)

            for b in range(n_bins):
                selected = np.where(indices == b + 1)[0]
                if len(selected) > 0:
                    bin_accuracies[b] += np.sum(true_labels[selected] == pred_labels[selected])
                    bin_confidences[b] += np.sum(confidences[selected])
                    bin_counts[b] += len(selected)

        # Divide each bin by its bin count and avoid division by zero!
        bin_accuracies /= np.where(bin_counts > 0, bin_counts, 1)
        bin_confidences /= np.where(bin_counts > 0, bin_counts, 1)
        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        mce = np.max(gaps)

        return {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce
        }

class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.process_time()
        return self

    def __exit__(self, *args):
        if self.device == 'cuda':
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start.elapsed_time(self.end) / 1000
        else:
            self.end = time.process_time()
            self.elapsed_time = self.end - self.start


class Sampler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __enter__(self):
        if isinstance(self.optimizer, NoisyOptimizer):
            self.optimizer._stash_param_averages()

    def __exit__(self, *args):
        if isinstance(self.optimizer, NoisyOptimizer):
            self.optimizer._restore_param_averages()
