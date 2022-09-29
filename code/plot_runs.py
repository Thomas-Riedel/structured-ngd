import argparse
import sys
import datetime
import pickle
import re
import os
import itertools
from typing import Union, List, Tuple

import torch.nn as nn
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from optimizers.rank_k_cov import *
from torch.optim import Adam
from reliability_diagrams import reliability_diagram


def parse_args() -> dict:
    """Parse command line arguments.

    :return: args_dict, dict, parsed arguments from command line
    """
    parser = argparse.ArgumentParser(description='Run noisy optimizers with parameters.')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR10")
    parser.add_argument('-m', '--model', type=str, default="resnet18")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=str, default='1e-1')
    parser.add_argument('--k', type=str, default='0')
    parser.add_argument('--mc_samples', type=str, default='1')
    parser.add_argument('--structure', type=str, default='rank_cov')
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--momentum_grad', type=float, default=0.6)
    parser.add_argument('--momentum_prec', type=float, default=0.999)
    parser.add_argument('--prior_precision', type=float, default=0.4)
    parser.add_argument('--damping', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('-s', '--data_split', type=float, default=0.8)
    parser.add_argument('--n_bins', type=int, default=10)

    args = parser.parse_args(sys.argv[1:])
    args.lr, args.k, args.mc_samples = parse_vals(
        [args.lr, args.k, args.mc_samples],
        [float, int, int]
    )
    n = len(args.lr)
    args.structure = n * [args.structure]
    args.momentum_grad = n * [args.momentum_grad]
    args.momentum_prec = n * [args.momentum_prec]
    args.prior_precision = n * [args.prior_precision]
    args.damping = n * [args.damping]
    args.gamma = n * [args.gamma]

    args_dict = dict(
        epochs=args.epochs,
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        k=args.k,
        mc_samples=args.mc_samples,
        structure=args.structure,
        eval_every=args.eval_every,
        momentum_grad=args.momentum_grad,
        momentum_prec=args.momentum_prec,
        prior_precision=args.prior_precision,
        damping=args.damping,
        gamma=args.gamma,
        data_split=args.data_split,
        n_bins=args.n_bins
    )
    return args_dict


def parse_vals(args: List[str], types: List[type]) -> List[Union[int, float]]:
    """Parse string values from command line, separate and form into list.

    :param args: List[str], list of supplied command line arguments as strings
    :param types: List[type], list of specified types to parse supplied arguments to
    :return: result, List[Union[int, float]], list of parsed and separated arguments
    """
    def make_sequence(type):
        def f(val):
            if type == int:
                # argument can be given as "0to11step3" meaning it should be parsed into a list ranging from 0 to 11
                # including with a step size of 3, i.e. [0, 3, 6, 9]
                val_list = [type(x) for x in re.split(r'to|step', val)]
                if len(val_list) == 3:
                    start, end, step = val_list
                elif len(val_list) == 2:
                    start, end = val_list
                    step = 1
                elif len(val_list) == 1:
                    start = val_list[0]
                    end = start + 1
                    step = 1
                else:
                    raise ValueError("Supply your parameter as <start>to<end>step<step_size>")
                return list(range(start, end, step))
            else:
                return [type(val)]
        return f
    result = []
    for arg, type in zip(args, types):
        result.append(list(sorted(set(itertools.chain.from_iterable(map(make_sequence(type), arg.split(',')))))))
    max_length = np.max([len(x) for x in result])
    for i in range(len(result)):
        if len(result[i]) == 1:
            result[i] *= max_length
        elif len(result[i]) < max_length:
            raise ValueError()
    return result


def load_data(dataset: str, batch_size: int, split: float = 0.8) -> Tuple[DataLoader]:
    """Load dataset, prepare for ResNet training, and split into train. validation and test set

    :param dataset: str, dataset to be downloaded (one of MNIST, FashionMNIST, or CIFAR-10)
    :param batch_size: int, batch size for data
    :param split: float, split for training and validation data
    :return: (train_loader, val_loader, test_loader), Tuple[torch.utils.data.DataLoader], list of split and batched
        dataset
    """
    assert(0 < split < 1)

    # For ResNet, see https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])

    if dataset.lower() == "mnist":
        training_data = MNIST('data/mnist/train', download=True, train=True, transform=transform)
        test_data = MNIST('data/mnist/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "fmnist":
        training_data = FashionMNIST('data/fmnist/train', download=True, train=True, transform=transform)
        test_data = FashionMNIST('data/fmnist/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "cifar10":
        training_data = CIFAR10('data/cifar10/train', download=True, train=True, transform=transform)
        test_data = CIFAR10('data/cifar10/test', download=True, train=False, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset} not recognized! Choose one of [mnist, fmnist, cifar10]")

    indices = list(range(len(training_data)))
    np.random.shuffle(indices)
    split = int(split * len(training_data))
    train_sampler = SubsetRandomSampler(indices[:split])
    val_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(training_data, sampler=train_sampler, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(training_data, sampler=val_sampler, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader


def get_params(args: dict) -> dict:
    ngd = [
        dict(
            lr=lr, k=k, mc_samples=mc_samples, structure=structure,
            momentum_grad=momentum_grad, momentum_prec=momentum_prec,
            prior_precision=prior_precision, damping=damping, gamma=gamma
        )
        for (lr, k, mc_samples, structure, momentum_grad, momentum_prec, prior_precision, damping, gamma) in
        zip(
            args['lr'], args['k'], args['mc_samples'], args['structure'],
            args['momentum_grad'], args['momentum_prec'],
            args['prior_precision'], args['damping'], args['gamma']
        )
    ]
    adam = [dict(lr=lr) for lr in set(args['lr'])]
    params = dict(ngd=ngd, adam=adam)
    return params


def run(epochs: int, model: nn.Module, optimizers: List[Union[Adam, StructuredNGD]],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None,
        adam_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [],
        eval_every: int = 10) -> List[dict]:
    """Run a list of optimizers on data for multiple epochs using multiple hyperparameters and evaluate.

    :param epochs: int, number of epochs for training
    :param model: str, ResNet model for experiments
    :param optimizers: List[Union[Adam, StructuredNGD]], list of models to run experiments on
    :param train_loader: torch.utils.data.DataLoader, training data
    :param val_loader: torch.utils.data.DataLoader, validation data
    :param test_loader: torch.utils.data.DataLoader, test data
    :param adam_params: List[dict], hyperparameters for Adam
    :param ngd_params: List[dict], hyperparameters for StructuredNGD
    :param metrics: List[Callable], list of metrics to run on data for evaluation
    :param eval_every: int, after a certain number of iterations, running losses and metrics will be averaged and
        displayed
    :return: runs, List[dict], list of results
    """
    loss_fn = nn.CrossEntropyLoss()
    runs = []
    device = model.device

    for optim in optimizers:
        if optim is StructuredNGD:
            params = ngd_params
        else:
            params = adam_params
        for param in params:
            print(optim.__name__, param)
            model.init_weights()
            if optim is StructuredNGD:
                optimizer = optim(model.parameters(), len(train_loader.dataset), device=device, **param)
            else:
                optimizer = optim(model.parameters(), lr=param['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            epoch_times = []
            iter_times = []
            train_loss = []
            val_loss = []
            val_metrics = {}
            iter_losses = []
            iter_metrics = {}
            for metric in metrics:
                val_metrics[metric.__name__] = []
                iter_metrics[metric.__name__] = []

            for epoch in range(epochs):
                iter_loss, iter_metric, iter_time = model.train(train_loader, optimizer, epoch=epoch,
                                                                loss_fn=loss_fn, metrics=metrics,
                                                                eval_every=eval_every)
                if epoch == 0:
                    epoch_times.append(0)
                else:
                    epoch_times.append(np.sum(iter_time))
                iter_times += iter_time

                loss, metric = model.evaluate(val_loader, metrics=metrics)
                # Append single validation metric value epoch-wise
                for metric_key in metric.keys():
                    val_metrics[metric_key].append(metric[metric_key])
                val_loss.append(loss)

                iter_losses += iter_loss
                # Append multiple values iteration-wise
                for metric_key in iter_metric.keys():
                    iter_metrics[metric_key] += iter_metric[metric_key]

                loss, _ = model.evaluate(train_loader, metrics=[])
                train_loss.append(loss)
                scheduler.step()

            test_loss, test_metrics = model.evaluate(test_loader, metrics=metrics)
            iter_times[0] = 0.0
            iter_times = np.cumsum(iter_times)
            epoch_times = np.cumsum(epoch_times)
            name = type(optimizer).__name__
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            test_labels, test_preds, test_logits = model.collect(test_loader)

            run = dict(
                name=name,
                # optimizer=optimizer,
                params=param,
                iter_times=iter_times,
                epoch_times=epoch_times,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                test_loss=test_loss,
                test_metrics=test_metrics,
                test_labels=test_labels,
                test_preds=test_preds,
                test_logits=test_logits,
                iter_loss=iter_losses,
                iter_metrics=iter_metrics,
                timestamp=timestamp
            )
            runs.append(run)
            save_runs(run)
        print('Finished Training')
    return runs


def save_runs(runs: Union[dict, List[dict]]) -> None:
    """Save runs in runs folder.

    :param runs: List[dict], list of runs to be individually saved in runs folder
    """
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('runs'):
        os.mkdir('runs')

    for run in runs:
        with open(f"runs/{run['timestamp']}.pkl", 'wb') as f:
            pickle.dump(run, f)


def load_runs(dir: str = 'runs'):
    runs = []
    for run in os.listdir(dir):
        if run.endswith(".pkl"):
            with open(os.path.join(dir, run), 'rb') as f:
                runs.append(pickle.load(f))
    return runs


def plot_runs(runs: Union[dict, List[dict]]) -> None:
    """Plot runs and save in plots folder.

    :param runs: List[dict], list of runs to be plotted
    """
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('plots'):
        os.mkdir('plots')
    plot_loss(runs)
    plot_metrics(runs)
    plot_generalization_gap(runs)
    plot_reliability_diagram(runs)


def plot_loss(runs):
    # Plot training loss in terms of iterations and computation time
    plt.figure(figsize=(12, 8))
    for run in runs:
        label = run['name']
        if label == 'StructuredNGD':
            label += f" (k = {run['params']['k']})"
        plt.subplot(2, 1, 1)
        plt.plot(run['iter_loss'], label=label)

        # plt.ylim(bottom=0)
        plt.title('Training Loss w.r.t. Iterations')
        plt.xlabel('iterations')
        plt.yscale('log')

        plt.subplot(2, 1, 2)
        plt.plot(run['iter_times'], run['iter_loss'], label=label)

        # plt.ylim(bottom=0)
        plt.title('Training Loss w.r.t Time')
        plt.xlabel('time (s)')
        plt.legend()
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/iter_losses.pdf')

    # Plot validation loss over epochs and computation time
    plt.figure(figsize=(12, 8))
    for run in runs:
        label = run['name']
        if label == 'StructuredNGD':
            label += f" (k = {run['params']['k']})"
        plt.subplot(2, 1, 1)
        plt.plot(run['val_loss'], label=label)
        plt.title('Validation Loss')
        plt.xlabel('epochs')
        plt.ylim(bottom=0)

        plt.subplot(2, 1, 2)
        plt.plot(run['epoch_times'], run['val_loss'], label=label)
        plt.title('Validation Loss')
        plt.xlabel('time (s)')
        plt.ylim(bottom=0)
        plt.legend()

    plt.tight_layout()
    plt.savefig('plots/val_losses.pdf')


def plot_metrics(runs):
    # Plot metrics
    plt.figure(figsize=(12, 8))
    for run in runs:
        plt.subplot(2, 1, 1)
        label = run['name']
        if label == 'StructuredNGD':
            label += f" (k = {run['params']['k']})"
        for metric_key in run['val_metrics'].keys():
            plt.plot(run['val_metrics'][metric_key],
                     label=f"{label}; ({metric_key})")

        plt.title('Validation Metrics w.r.t. Epochs')
        plt.xlabel('epochs')
        plt.ylim(0, 1)

        plt.subplot(2, 1, 2)
        for metric_key in run['val_metrics'].keys():
            plt.plot(run['epoch_times'], run['val_metrics'][metric_key],
                     label=f"{label}; ({metric_key})")
        plt.title('Validation Metrics w.r.t. Time')
        plt.xlabel('time (s)')
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.savefig('plots/val_metrics.pdf')


def smooth(x):
    pass


def plot_generalization_gap(runs):
    for run in runs:
        plt.figure(figsize=(12, 8))
        label = run['name']
        if label == 'StructuredNGD':
            label += f" (k = {run['params']['k']})"

        # Plot generalization gap in terms wrt epochs
        min_index = np.argmin(run['val_loss'])
        ymax = np.concatenate((run['train_loss'], run['val_loss'])).max()
        plt.plot(run['train_loss'], label='Train Data')
        plt.plot(run['val_loss'], label='Validation Data')
        plt.vlines(min_index, ymin=0, ymax=ymax, colors='r', linestyle='dashed')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title(f"Generalization Gap ({label})")

        # # Plot generalization gap in terms wrt time
        # plt.plot(run['epoch_times'], run['train_loss'], label='Train Data')
        # plt.plot(run['epoch_times'], run['val_loss'], label='Validation Data')
        # plt.vlines(run['epoch_times'][min_index], ymin=0, ymax=ymax, colors='r', linestyle='dashed')
        # plt.xlabel('time (s)')
        # plt.ylabel('Loss')
        # plt.title('Generalization Gap')
        plt.legend()
        plt.savefig(f"plots/{run['timestamp']}_generalization_gap.pdf")


def plot_reliability_diagram(runs, n_bins=10):
    for run in runs:
        plt.figure(figsize=(12, 8))
        labels = run['test_labels']
        preds = run['test_preds']
        logits = run['test_logits']
        reliability_diagram(labels, preds, logits, num_bins=n_bins,
                            draw_ece=True, draw_mce=True,
                            draw_bin_importance="alpha", draw_averages=True,
                            figsize=(6, 6), dpi=100,
                            return_fig=True)
        plt.savefig(f"plots/{run['timestamp']}_reliability_diagram.pdf")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    runs = load_runs()
    plot_runs(runs)