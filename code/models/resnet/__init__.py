import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from optimizers.noisy_optimizer import *
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchsummary import summary

from typing import Union, Tuple, List, Callable
import time


class Model(nn.Module):
    def __init__(self, model_type: str = 'resnet20',
                 num_classes: int = 10, input_shape=(3, 32, 32),
                 device: str = None) -> None:
        """torch.nn.Module class for ResNet

        :param model_type: str, specify what ResNet model to use
        :param num_classes: int, number of classes in dataset for classification
        :param device: str, torch device to run operations on (GPU or CPU)
        """
        super(Model, self).__init__()

        model_type = model_type.lower()
        model_types_imagenet = ['resnet' + str(n) for n in [18, 34, 50, 101, 152]]
        model_types_cifar10 = ['resnet' + str(n) for n in [20, 32, 44, 56, 110, 1202]]
        if model_type in model_types_cifar10:
            print(f"Using {model_type} designed for CIFAR-10.")
            self.model = eval(f"{model_type}(num_classes=num_classes).to(device)")
            # torch.hub.load('pytorch/vision:v0.10.0',
            #                model_type, pretrained=False,
            #                num_classes=num_classes).to(device)
        elif model_type in model_types_imagenet:
            print(f"Using {model_type} designed for ImageNet.")
            self.model = eval(f"{model_type}(pretrained=False, num_classes=num_classes).to(device)")
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
            model_types = model_types_imagenet + model_types_cifar10
            raise ValueError(f"Model type {model_type} not recognized! "
                             f"Choose one of {model_types}.")

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
            start = time.time()
            loss, preds = optimizer.step(closure)
            end = time.time()
            epoch_time += end - start
            # print(loss.item())

            # Record losses and metrics
            running_loss += loss.item()
            epoch_loss += loss.item()
            for metric in metrics:
                running_metrics[metric.__name__] += metric(preds, labels).detach().cpu().numpy()
                epoch_metrics[metric.__name__] += metric(preds, labels).detach().cpu().numpy()

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
                 loss_fn: Callable = nn.CrossEntropyLoss()) -> Tuple[float, dict]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals), Tuple[float, dict], tuple of loss and specified metrics on validation set
        """
        if data_loader is None:
            return None, None

        # Set model to evaluation mode!
        self.model.eval()

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

    @torch.no_grad()
    def compute_calibration(self, data_loader, n_bins=10):
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
            confidences = torch.max(F.softmax(self(images), dim=1), axis=1)
            pred_labels = confidences.indices
            confidences = confidences.values

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


# Code by https://github.com/akamaster/pytorch_resnet_cifar10
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10, *args, **kwargs):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)
