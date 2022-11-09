import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

import time
from typing import Union, Tuple, List, Callable

from optimizers.noisy_optimizer import *
from metrics import *
from models import *
from util import *


class Model(nn.Module):
    def __init__(self, model_type: str = 'ResNet20', num_classes: int = 10, input_shape=(3, 32, 32),
                 device: str = None, bnn: bool = False, dropout_layers: Union[None, str] = None, p: float = 0.2) -> None:
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
            # MC Dropout only implemented for ResNet models!
            self.model = eval(f"{model_type}(num_classes=num_classes, dropout_layers=dropout_layers, p=p).to(device)")
        else:
            print(f"Using {model_type}.")
            # MC Dropout only implemented for ResNet models!
            self.model = eval(f"{model_type}(num_classes=num_classes, dropout_layers=dropout_layers, p=p).to(device)")

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mc_dropout = False
        if not dropout_layers is None:
            self.mc_dropout = True
        if bnn:
            const_bnn_prior_parameters = {
                "prior_mu": 0.0,
                "prior_sigma": 1.0,
                "posterior_mu_init": 0.0,
                "posterior_rho_init": -3.0,
                "type": "Reparameterization",
                "moped_enable": False,
                "moped_delta": 0.5,
            }
            dnn_to_bnn(self.model, const_bnn_prior_parameters)
            self.model = self.model.to(device)
        self.bnn = bnn
        self.init_weights()
        self.num_classes = num_classes
        self.device = device

        self.__name__ = model_type
        self.summary = summary(self.model, input_shape)
        self.num_params = np.sum(p.numel() for p in self.parameters())

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through model.

        :param x: torch.tensor, argument for forward pass
        :return: output, Tensor, model output
        """
        return self.model(x)

    def train(self, data_loader: DataLoader, optimizer: Union[Adam, NoisyOptimizer],
              epoch: int = 0, metrics: List[Callable] = [], eval_every: int = 100,
              loss_fn: Callable = nn.NLLLoss()) -> Tuple[List[float], dict, List[float]]:
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
        if self.bnn or self.mc_dropout:
            assert(not isinstance(optimizer, NoisyOptimizer))
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

            if self.bnn:
                def closure():
                    optimizer.zero_grad()
                    logits = self(images)
                    preds = F.log_softmax(logits, dim=-1)
                    kl = get_kl_loss(self.model)
                    ce_loss = loss_fn(preds, labels)
                    loss = ce_loss + kl / len(data_loader.dataset)
                    loss.backward()
                    return ce_loss, logits
            else:
                def closure():
                    optimizer.zero_grad()
                    logits = self(images)
                    preds = F.log_softmax(logits, dim=-1)
                    loss = loss_fn(preds, labels)
                    loss.backward()
                    return loss, logits

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
                 loss_fn: Callable = nn.NLLLoss(),
                 optimizer=None, mc_samples: int = 1, n_bins: int = 10) -> Tuple[float, dict, dict, dict]:
        """Evaluate data on loss function and metrics.

        :param data_loader: torch.utils.data.DataLoader, validation or test dataset
        :param metrics: List[Callable], list of metrics to run for evaluation
        :param loss_fn: Callable, loss function to optimize
        :return: (loss, metric_vals, bin_data), Tuple[float, dict, dict],
            tuple of loss and specified metrics on validation set
        """
        if not isinstance(optimizer, NoisyOptimizer) and not (self.bnn or self.mc_dropout):
            mc_samples = 1
        assert(mc_samples >= 1)
        # Set model to evaluation mode!
        self.model.eval()

        # If noisy optimizer (NGD) is used, sample weights once, perform full forward pass and repeat for all MC samples
        logits = []
        labels_list = []
        with Sampler(optimizer):
            for mc_sample in range(mc_samples):
                mc_logits = []
                if isinstance(optimizer, NoisyOptimizer):
                    optimizer._sample_weight()
                for i, data in enumerate(data_loader):
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    mc_logits.append(self(images))
                    if mc_sample == 0:
                        labels_list.append(labels)
                mc_logits = torch.cat(mc_logits)
                logits.append(mc_logits)
        logits = torch.stack(logits)
        labels = torch.cat(labels_list)
        loss = loss_fn(F.log_softmax(logits, dim=-1).mean(0), labels).item()

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
