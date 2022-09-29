import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from optimizers.noisy_optimizer import *

from typing import Union, Tuple, List, Callable
import time


class ResNet(nn.Module):
    def __init__(self, model_type: str = 'resnet18',
                 num_classes: int = 10,
                 device: str = None) -> None:
        """torch.nn.Module class for ResNet

        :param model_type: str, specify what ResNet model to use
        :param num_classes: int, number of classes in dataset for classification
        :param device: str, torch device to run operations on (GPU or CPU)
        """
        super(ResNet, self).__init__()

        model_types = ['resnet' + str(n) for n in [18, 34, 50, 101, 152]]
        if model_type.lower() in model_types:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                        model_type, pretrained=False,
                                        num_classes=num_classes).to(device)
        else:
            # self.model = nn.Sequential(
            #     nn.Conv2d(input_shape[1], dim // 4, (3, 3)),
            #     nn.BatchNorm2d(dim // 4),
            #     nn.ReLU(0.2),
            #     nn.MaxPool2d(2, 2),
            #     nn.Conv2d(dim // 4, dim // 2, (3, 3)),
            #     nn.BatchNorm2d(dim // 2),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, 2),
            #     nn.Conv2d(dim // 2, dim, (3, 3)),
            #     nn.BatchNorm2d(dim),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, 2),
            #     nn.Flatten(),
            #     nn.Linear(dim, dim // 2),
            #     nn.ReLU(),
            #     nn.Linear(dim // 2, dim // 4),
            #     nn.ReLU(),
            #     nn.Linear(dim // 4, num_classes),
            #     nn.Softmax(dim=1)
            # )
            raise ValueError(f"Model type {model_type} not recognized! "
                             f"Choose one of {model_types}.")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_weights()
        self.num_classes = num_classes
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through model.

        :param x: torch.tensor, argument for forward pass
        :return: output, Tensor, model output
        """
        return self.model(x)

    def train(self, data_loader: DataLoader, optimizer: Union[Adam, NoisyOptimizer],
              epoch: int = 0, metrics: List[Callable] = [], eval_every: int = 10,
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
        epoch_loss = 0.0
        iter_loss = []
        iter_metrics = {}
        iter_time = []
        running_loss = 0.0
        running_metrics = {}
        for metric in metrics:
            iter_metrics[metric.__name__] = []
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
            start = time.time()
            loss, preds = optimizer.step(closure)
            end = time.time()
            iter_time.append(end - start)
            # print(loss.item())

            # Record losses and metrics
            iter_loss.append(loss.item())
            running_loss += loss.item()
            epoch_loss += loss.item()
            for metric in metrics:
                running_metrics[metric.__name__] += metric(preds, labels).detach().cpu().numpy()
                iter_metrics[metric.__name__].append(metric(preds, labels).item())

            if i % eval_every == (eval_every - 1):
                print("===========================================")
                print(f"[{epoch + 1}, {i + 1}] Total loss: {running_loss / eval_every:.3f}")
                running_loss = 0.0
                for metric in metrics:
                    name = metric.__name__
                    print(f"\t{name}: {running_metrics[name] / eval_every:.3f}")
                    running_metrics[name] = 0.0
                print("===========================================")
        return iter_loss, iter_metrics, iter_time

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, metrics: List[Callable] = [],
                 loss_fn: Callable = nn.CrossEntropyLoss()) -> Tuple[float, dict]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals), Tuple[float, dict], tuple of loss and specified metrics on validation set
        """
        if data_loader is None:
            return None, None
        loss = 0.0
        metric_vals = {}
        for metric in metrics:
            metric_vals[metric.__name__] = 0.0
        for i, data in enumerate(data_loader):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self(images)
            loss += loss_fn(preds, labels).item()
            for metric in metrics:
                metric_vals[metric.__name__] += metric(preds, labels).detach().cpu().numpy()

            # Write to TensorBoard
            # writer.add_scalar("Loss", loss, counter)

        for metric in metrics:
            metric_vals[metric.__name__] /= len(data_loader)
        loss /= len(data_loader)
        print(loss, metric_vals)
        return loss, metric_vals

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

    def compute_calibration(self, data_loader, num_bins=10):
        """Collects predictions into bins used to draw a reliability diagram.
            Adapted code snippet by https://github.com/hollance/reliability-diagrams

        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            num_bins: number of bins

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
        assert(num_bins > 0)

        bin_accuracies = np.zeros(num_bins, dtype=float)
        bin_confidences = np.zeros(num_bins, dtype=float)
        bin_counts = np.zeros(num_bins, dtype=int)
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        avg_acc = 0.0
        avg_conf = 0.0
        ece = 0.0
        mce = 0.0

        for data in data_loader:
            image, true_label = data
            image.to(self.device)
            true_label.to(self.device)
            confidence = self(image)
            pred_label = torch.argmax(confidence, axis=1)

            true_label = true_label.detach().cpu().numpy()
            confidence = confidence.detach().cpu().numpy()
            pred_label = pred_label.detach().cpu().numpy()

            bin_size = 1.0 / num_bins
            indices = np.digitize(confidence, bins, right=True)

            bin_accuracy = np.zeros(num_bins, dtype=float)
            bin_confidence = np.zeros(num_bins, dtype=float)
            bin_count = np.zeros(num_bins, dtype=int)

            for b in range(num_bins):
                selected = np.where(indices == b + 1)[0]
                if len(selected) > 0:
                    bin_accuracy[b] = np.mean(true_label[selected] == pred_label[selected])
                    bin_confidence[b] = np.mean(confidence[selected])
                    bin_count[b] = len(selected)

            bin_accuracies += bin_accuracy
            bin_confidences += bin_confidence
            bin_counts += bin_count

            avg_acc += np.sum(bin_accuracy * bin_count) / np.sum(bin_count)
            avg_conf += np.sum(bin_confidence * bin_count) / np.sum(bin_count)

            gaps = np.abs(bin_accuracy - bin_confidence)
            ece += np.sum(gaps * bin_count) / np.sum(bin_count)
            mce += np.max(gaps)

        n = len(data_loader)
        bin_accuracies /= n
        bin_confidences /= n
        avg_acc /= n
        avg_conf /= n
        ece /= n
        mce /= n

        return { "accuracies": bin_accuracies,
                 "confidences": bin_confidences,
                 "counts": bin_counts,
                 "bins": bins,
                 "avg_accuracy": avg_acc,
                 "avg_confidence": avg_conf,
                 "expected_calibration_error": ece,
                 "max_calibration_error": mce}

    def init_weights(self, seed: int = 42) -> None:
        """Initialize weights.

        :param seed: int, seed for random initialization
        """
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_(0, 1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
