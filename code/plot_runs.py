import argparse
import os.path
import sys
import datetime
import re
import itertools
from typing import Any

import torch.nn as nn
import torch.utils.data
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CIFAR100, STL10, SVHN, ImageNet
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from optimizers.rank_k_cov import *
from torch.optim import *
from reliability_diagrams import reliability_diagram
from corruption import *
from metrics import *
from network import *

CIFAR_TRANSFORM_AUGMENTED = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))]
)

CIFAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))]
)


def parse_args() -> dict:
    """Parse command line arguments.

    :return: args_dict, dict, parsed arguments from command line
    """
    parser = argparse.ArgumentParser(description='Run methods with parameters.')
    parser.add_argument('-o', '--optimizer', type=str, default='StructuredNGD',
                        help='Optimizer, one of Adam, SGD, StructuredNGD (capitalization matters!, default: StructuredNGD)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train models on data (default: 1)')
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR10",
                        help='Dataset for training, one of CIFAR10, CIFAR100, MNIST, FashionMNIST (default: CIFAR10)')
    parser.add_argument('-m', '--model', type=str, default="ResNet20",
                        help='ResNet model (default: ResNet18)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for data loaders (default: 128)')
    parser.add_argument('--lr', type=str, default='1e-3',
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--k', type=str, default='0',
                        help='Rank parameter for StructuredNGD (default: 0)')
    parser.add_argument('--mc_samples', type=str, default='1',
                        help='Number of MC samples during training (default: 1)')
    parser.add_argument('--structure', type=str, default='rank_cov',
                        help='Covariance structure (default: rank_cov)')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Frequency of summary statistics printing during training (default: 100)')
    parser.add_argument('--momentum_grad', type=float, default=0.9,
                        help='First moment strength (default: 0.9)')
    parser.add_argument('--momentum_prec', type=float, default=0.999,
                        help='Second moment strength (default: 0.999)')
    parser.add_argument('--prior_precision', type=float, default=0.4,
                        help='Spherical prior precision (default: 0.4)')
    parser.add_argument('--damping', type=float, default=0.1,
                        help='Damping strength for matrix inversion (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Regularization parameter in ELBO (default: 1.0)')
    parser.add_argument('-s', '--data_split', type=float, default=0.8,
                        help='Data split for training and validation set (default: 0.8)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of bins for reliability diagrams (default: 10)')
    parser.add_argument('--mc_samples_eval', type=int, default=32,
                        help='Number of MC samples during evaluation (default: 32)')
    parser.add_argument('--baseline', type=str, default='SGD',
                        help='Baseline optimizer for comparison of corrupted data')

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
    args.optimizer = eval(args.optimizer)

    args_dict = dict(
        optimizer=args.optimizer,
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
        n_bins=args.n_bins,
        mc_samples_eval=args.mc_samples_eval,
        baseline=args.baseline
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


def load_data(dataset: str, batch_size: int, split: float = 0.8, pad_size: int = 4) -> \
        Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Load dataset, prepare for ResNet training, and split into train. validation and test set

    :param dataset: str, dataset to be downloaded (one of MNIST, FashionMNIST, or CIFAR-10)
    :param batch_size: int, batch size for data
    :param split: float, split for training and validation data
    :return: (train_loader, val_loader, test_loader), Tuple[torch.utils.data.DataLoader], list of split and batched
        dataset
    """
    assert (0 < split < 1)

    if dataset.lower().startswith('cifar'):
        transform_augmented = CIFAR_TRANSFORM_AUGMENTED
        transform = CIFAR_TRANSFORM
    else:
        transform_augmented = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=pad_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)), ]
        )

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        )

    if dataset.lower() == "mnist":
        training_data = MNIST('data/MNIST/train', download=True, train=True, transform=transform_augmented)
        test_data = MNIST('data/MNIST/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "fmnist":
        training_data = FashionMNIST('data/FasionMNIST/train', download=True, train=True, transform=transform_augmented)
        test_data = FashionMNIST('data/FashionMNIST/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "cifar10":
        training_data = CIFAR10('data/CIFAR10/train', download=True, train=True, transform=transform_augmented)
        test_data = CIFAR10('data/CIFAR10/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "cifar100":
        training_data = CIFAR100('data/CIFAR100/train', download=True, train=True, transform=transform_augmented)
        test_data = CIFAR100('data/CIFAR100/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "stl10":
        training_data = STL10('data/STL10/train', download=True, split='train', transform=transform_augmented)
        test_data = STL10('data/STL10/test', download=True, split='test', transform=transform)
    elif dataset.lower() == "imagenet":
        training_data = ImageNet('data/ImageNet/train', download=True, split='train', transform=transform_augmented)
        test_data = ImageNet('data/ImageNet/test', download=True, split='val', transform=transform)
    elif dataset.lower() == "svhn":
        training_data = SVHN('data/SVHN/train', download=True, split='train', transform=transform_augmented)
        test_data = SVHN('data/SVHN/test', download=True, split='test', transform=transform)
    else:
        raise ValueError(f"Dataset {dataset} not recognized! Choose one of "
                         f"[mnist, fmnist, cifar10, cifar100, stl10, imagenet, svhn]")

    indices = list(range(len(training_data)))
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = int(split * len(training_data))
    train_sampler = SubsetRandomSampler(indices[:split])
    val_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(
        training_data, sampler=train_sampler, batch_size=batch_size, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        training_data, sampler=val_sampler, batch_size=batch_size, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_params(args: dict, baseline='SGD', add_weight_decay=True, n=1) -> dict:
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
    if baseline == 'Adam':
        momentum = dict(betas=(0.9, 0.999))
    elif baseline == 'SGD':
        momentum = dict(momentum=0.9, nesterov=True)
    baseline = [dict(lr=lr, weight_decay=add_weight_decay * prior_precision / n, **momentum)
            for lr, prior_precision in zip(set(args['lr']), set(args['prior_precision']))]
    params = dict(ngd=ngd, baseline=baseline)
    return params


def record_loss_and_metrics(losses, metrics, loss, metric):
    if len(metrics.keys()) == 0:
        for key in metric:
            metrics[key] = []
    assert (metrics.keys() == metric.keys())
    losses.append(loss)
    for key in metric.keys():
        metrics[key].append(metric[key])
    return losses, metrics


def run_experiments(epochs: int, methods: List[str], model: nn.Module, optimizer: List[Union[Optimizer, StructuredNGD]],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, baseline: str = 'SGD',
        baseline_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [accuracy],
        eval_every: int = 100, n_bins: int = 10, mc_samples: int = 32) -> List[dict]:
    """Run an optimizer on data for multiple epochs using multiple hyperparameters and evaluate.

    :param epochs: int, number of epochs for training
    :param model: str, ResNet model for experiments
    :param optimizer: List[Union[baseline, StructuredNGD]], list of models to run experiments on
    :param train_loader: torch.utils.data.DataLoader, training data
    :param val_loader: torch.utils.data.DataLoader, validation data
    :param test_loader: torch.utils.data.DataLoader, test data
    :param baseline_params: List[dict], hyperparameters for baseline
    :param ngd_params: List[dict], hyperparameters for StructuredNGD
    :param metrics: List[Callable], list of metrics to run on data for evaluation
    :param eval_every: int, after a certain number of iterations, running losses and metrics will be averaged and
        displayed
    :return: runs, List[dict], list of results
    """
    if isinstance(model, (TempScaling, DeepEnsemble, HyperDeepEnsemble)):
        return evaluate(methods[0], model, train_loader, val_loader, test_loader, baseline, metrics, n_bins, mc_samples)
    else:
        return train_and_evaluate(epochs, methods, model, optimizer, train_loader, val_loader, test_loader,
                                  baseline, baseline_params, ngd_params, metrics, eval_every, n_bins, mc_samples)


def evaluate(method: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
             baseline: str = 'SGD', metrics: List[Callable] = [accuracy], n_bins: int = 10, mc_samples: int = 32):
    if isinstance(model, TempScaling):
        model.set_temperature(val_loader)
    runs = []
    dataset = train_loader.dataset.root.split('/')[1]

    train_loss = []
    train_metrics = {}
    loss, metric, _ = model.evaluate(train_loader, metrics=metrics, n_bins=n_bins)
    train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

    val_loss = []
    val_metrics = {}
    loss, metric, _ = model.evaluate(val_loader, metrics=metrics, n_bins=n_bins)
    val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

    test_loss, test_metrics, bin_data = model.evaluate(test_loader, metrics=metrics, n_bins=n_bins)
    clean_results = dict(
        test_loss=test_loss,
        test_metrics=test_metrics,
        bin_data=bin_data
    )
    corrupted_results = get_corrupted_results(dataset, model, baseline, method, baseline, metrics,
                                              clean_results, mc_samples, n_bins)

    param = dict()
    num_epochs = np.sum([run['num_epochs'] + 1 for run in runs])
    epoch_times = [None]
    total_time = np.sum([run['epoch_times'][-1] for run in runs])
    avg_time_per_epoch = total_time / num_epochs
    optimizer_name = baseline
    model_name = model.__name__
    num_params = model.num_params
    model_summary = None

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    run = dict(
        optimizer_name=optimizer_name,
        model_name=model_name,
        method=method,
        dataset=dataset,
        num_params=num_params,
        model_summary=model_summary,
        params=param,
        epoch_times=epoch_times,
        num_epochs=num_epochs,
        avg_time_per_epoch=avg_time_per_epoch,
        total_time=total_time,
        train_loss=train_loss,
        train_metrics=train_metrics,
        val_loss=val_loss,
        val_metrics=val_metrics,
        test_loss=test_loss,
        test_metrics=test_metrics,
        bin_data=bin_data,
        corrupted_results=corrupted_results,
        timestamp=timestamp
    )
    runs.append(run)
    save_runs(run)
    return runs


def train_and_evaluate(epochs: int, methods: List[str], model: nn.Module, optim: Union[Optimizer, StructuredNGD],
                       train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, baseline: str = 'SGD',
                       baseline_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [accuracy],
                       eval_every: int = 100, n_bins: int = 10, mc_samples: int = 32):
    loss_fn = nn.CrossEntropyLoss()
    runs = []
    dataset = train_loader.dataset.root.split('/')[1]
    if optim is StructuredNGD:
        params = ngd_params
    else:
        params = baseline_params
    for i, param in enumerate(params):
        print(optim.__name__, param)
        early_stopping = EarlyStopping(patience=16)
        if optim is StructuredNGD:
            seed = 42
            optimizer = optim(model.parameters(), len(train_loader.dataset), **param)
        else:
            seed = None
            optimizer = optim(model.parameters(), **param)
        model.init_weights(seed)
        run = load_run(dataset, model, baseline)
        optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
        if (dataset.lower() in ['cifar10', 'cifar100']) and (optimizer_name != baseline) and (run is None):
            raise RuntimeError(f"Baseline {baseline} does not exist for this dataset and model!"
                               f"Please first run the script with this baseline for the dataset and model.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8)

        # Initialize computation times, losses and metrics for recording
        loss, metric, _ = model.evaluate(train_loader, metrics=metrics)
        epoch_times = [0.0]
        train_loss = []
        train_metrics = {}
        train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

        loss, metric, _ = model.evaluate(val_loader, metrics=metrics)
        val_loss = []
        val_metrics = {}
        val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

        early_stopping.on_train_begin()
        for epoch in range(epochs):
            # Train for one epoch
            loss, metric, comp_time = model.train(train_loader, optimizer, epoch=epoch,
                                                  loss_fn=loss_fn, metrics=metrics,
                                                  eval_every=eval_every)
            # Record epoch times, loss and metrics for epoch
            epoch_times.append(comp_time)
            train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

            # Record loss and metrics for epoch
            loss, metric, _ = model.evaluate(val_loader, metrics=metrics)
            val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

            scheduler.step(loss)
            early_stopping.on_epoch_end(epoch, loss)
            if early_stopping.stop_training:
                break
        early_stopping.on_train_end()

        test_loss, test_metrics, bin_data = model.evaluate(test_loader, metrics=metrics, optimizer=optimizer,
                                                           mc_samples=mc_samples, n_bins=n_bins)
        clean_results = dict(
            test_loss=test_loss,
            test_metrics=test_metrics,
            bin_data=bin_data
        )
        corrupted_results = get_corrupted_results(dataset, model, optimizer, methods[i], baseline, metrics,
                                                  clean_results, mc_samples, n_bins)

        num_epochs = epoch + 1
        epoch_times = np.cumsum(epoch_times)
        total_time = epoch_times[-1]
        avg_time_per_epoch = epoch_times[-1] / num_epochs
        optimizer_name = type(optimizer).__name__
        model_name = model.__name__
        num_params = model.num_params
        model_summary = model.summary

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        run = dict(
            optimizer_name=optimizer_name,
            model_name=model_name,
            method=methods[i],
            dataset=dataset,
            num_params=num_params,
            model_summary=model_summary,
            params=param,
            epoch_times=epoch_times,
            num_epochs=num_epochs,
            avg_time_per_epoch=avg_time_per_epoch,
            total_time=total_time,
            train_loss=train_loss,
            train_metrics=train_metrics,
            val_loss=val_loss,
            val_metrics=val_metrics,
            test_loss=test_loss,
            test_metrics=test_metrics,
            bin_data=bin_data,
            corrupted_results=corrupted_results,
            timestamp=timestamp
        )
        runs.append(run)
        save_runs(run)
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"checkpoints/{timestamp}.pt")
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


def plot_runs(runs: Union[dict, List[dict]]) -> None:
    """Plot runs and save in plots folder.

    :param runs: List[dict], list of runs to be plotted
    """
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('plots'):
        os.mkdir('plots')
    runs = [run for run in runs if run['model_name'].startswith('ResNet')]# and
            # (run['params'].get('structure') in [None, 'arrowhead'])]
    # runs = [run for run in runs if (run['num_epochs'] < 250) or
    #         (run['optimizer_name'] == 'SGD') or (run['dataset'] == 'stl10')]
    make_csv(runs)

    # plot_results_wrt_parameters(runs)
    # plot_loss(runs)
    # plot_metrics(runs)
    # plot_generalization_gap(runs)
    # plot_reliability_diagram(runs)
    plot_corrupted_data(runs)


def plot_loss(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    # Plot training and validation loss in terms of epochs and computation time
    plt.figure()
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
        if not os.path.exists(f"plots/{dataset}/losses"):
            os.makedirs(f"plots/{dataset}/losses", exist_ok=True)
        gammas = [run['params'].get('gamma') for run in runs if run['dataset'] == dataset]
        mc_samples = [run['params'].get('mc_samples') for run in runs if run['dataset'] == dataset]
        gammas = list(set([x for x in gammas if x is not None]))
        mc_samples = list(set([x for x in mc_samples if x is not None]))
        for gamma in gammas:
            for run in runs:
                if run['dataset'] != dataset:
                    continue
                if run['optimizer_name'].startswith('StructuredNGD') and (run['params']['gamma'] != gamma):
                    continue
                print(run['optimizer_name'])
                label = run['optimizer_name'] \
                    if not run['optimizer_name'].startswith('StructuredNGD') else \
                    f"k = {run['params']['k']}, M = {run['params']['mc_samples']}"
                train_label = f"Train Loss ({label})"
                val_label = f"Validation Loss ({label})"
                plt.subplot(2, 1, 1)
                plt.plot(run['train_loss'], label=train_label)
                plt.plot(run['val_loss'], label=val_label)

                # plt.ylim(bottom=0)
                plt.title(f"Loss w.r.t. Epochs (gamma = {gamma})")
                plt.xlabel('epochs')
                plt.yscale('log')

                plt.subplot(2, 1, 2)
                plt.plot(run['epoch_times'], run['train_loss'], label=train_label)
                plt.plot(run['epoch_times'], run['val_loss'], label=val_label)

                plt.title(f"Loss w.r.t Time (gamma = {gamma})")
                plt.xlabel('time (s)')
                plt.yscale('log')
            reorderLegend()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(f"plots/{dataset}/losses/gamma_{gamma}.pdf")
            plt.show()

        for samples in mc_samples:
            for run in runs:
                if run['dataset'] != dataset:
                    continue
                if run['optimizer_name'].startswith('StructuredNGD') and (run['params']['mc_samples'] != samples):
                    continue
                label = run['optimizer_name'] \
                    if not run['optimizer_name'].startswith('StructuredNGD') else \
                    f"k = {run['params']['k']}, gamma = {run['params']['gamma']}"
                train_label = f"Train Loss ({label})"
                val_label = f"Validation Loss ({label})"

                plt.subplot(2, 1, 1)
                plt.plot(run['train_loss'], label=train_label)
                plt.plot(run['val_loss'], label=val_label)

                # plt.ylim(bottom=0)
                plt.title(f"Loss w.r.t. Epochs (M = {samples})")
                plt.xlabel('epochs')
                # plt.yscale('log')

                plt.subplot(2, 1, 2)
                plt.plot(run['epoch_times'], run['train_loss'], label=train_label)
                plt.plot(run['epoch_times'], run['val_loss'], label=val_label)

                plt.title(f"Loss w.r.t Time (M = {samples})")
                plt.xlabel('time (s)')
                # plt.yscale('log')
            reorderLegend()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(f"plots/{dataset}/losses/mc_samples_{samples}.pdf")
            plt.show()


def plot_metrics(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
        if not os.path.exists(f"plots/{dataset}/metrics"):
            os.makedirs(f"plots/{dataset}/metrics", exist_ok=True)
        plt.figure()
        for run in runs:
            if run['dataset'] != dataset:
                continue
            # Plot metrics for training validation set wrt epochs
            optimizer = run['optimizer_name']
            plt.subplot(2, 1, 1)
            for key in run['val_metrics'].keys():
                train_metric = 100 * np.array(run['train_metrics'][key])
                val_metric = 100 * np.array(run['val_metrics'][key])
                plt.plot(train_metric,
                         label=f"Train Data ({key}; {optimizer})")
                plt.plot(val_metric,
                         label=f"Validation Data ({key}; {optimizer})")

            plt.title(f"Metrics w.r.t. Epochs ({dataset.upper()})")
            plt.xlabel('epochs')
            # plt.ylim(0, 1)
            # plt.legend()

            # Plot metrics for training validation set wrt computation time
            # plt.figure()
            plt.subplot(2, 1, 2)
            for key in run['val_metrics'].keys():
                train_metric = 100 * np.array(run['train_metrics'][key])
                val_metric = 100 * np.array(run['val_metrics'][key])
                plt.plot(run['epoch_times'], train_metric,
                         label=f"Train Data ({key}; {optimizer})")
                plt.plot(run['epoch_times'], val_metric,
                         label=f"Validation Data ({key}; {optimizer})")
            plt.title(f"Metrics w.r.t. Time ({dataset.upper()})")
            plt.xlabel('time (s)')
            # plt.ylim(0, 1)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}/metrics/metrics_{dataset}.pdf")


def plot_generalization_gap(runs: Union[dict, List[dict]]) -> None:
    """Plot generalization gap for each run.

    :param runs:
    :return:
    """
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        dataset = run['dataset']
        if not os.path.exists(f"plots/{dataset}/generalization_gap"):
            os.makedirs(f"plots/{dataset}/generalization_gap", exist_ok=True)
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            label = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
        else:
            label = run['optimizer_name']

        # Plot generalization gap in terms wrt epochs
        plt.figure()
        min_index = np.argmin(run['val_loss'])
        ymax = np.concatenate((run['train_loss'], run['val_loss'])).max()
        plt.plot(run['train_loss'], label='Train Data')
        plt.plot(run['val_loss'], label='Validation Data')
        plt.vlines(min_index, ymin=0, ymax=ymax, colors='r', linestyle='dashed')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title(f"Generalization Gap ({label})")

        # # Plot generalization gap in terms wrt time
        # plt.figure()
        # plt.plot(run['epoch_times'], run['train_loss'], label='Train Data')
        # plt.plot(run['epoch_times'], run['val_loss'], label='Validation Data')
        # plt.vlines(run['epoch_times'][min_index], ymin=0, ymax=ymax, colors='r', linestyle='dashed')
        # plt.xlabel('time (s)')
        # plt.ylabel('Loss')
        # plt.title('Generalization Gap')
        # plt.legend()
        # plt.savefig(f"plots/{run['timestamp']}_generalization_gap.pdf")

        plt.legend()
        plt.savefig(f"plots/{dataset}/generalization_gap/{label}.pdf")
        plt.show()


def plot_reliability_diagram(runs: Union[dict, List[dict]]) -> None:
    """Plot reliability diagram for each run and save in plots folder.
    The function 'reliability_diagram' is adapted from https://github.com/hollance/reliability-diagrams

    :param runs:
    """
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        dataset = run['dataset']
        if not os.path.exists(f"plots/{dataset}/reliability_diagrams"):
            os.makedirs(f"plots/{dataset}/reliability_diagrams", exist_ok=True)
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
            optimizer_name = f"NGD ({structure}, k = {params['k']}, " \
                             f"M = {params['mc_samples']}, gamma = {params['gamma']})"
        else:
            optimizer = run['optimizer_name']
            optimizer_name = run['optimizer_name']
        model = run['model_name']
        title = f"Reliability Diagram on {dataset.upper()} using {model}\n{optimizer}"
        plt.figure()
        bin_data = run['bin_data']
        reliability_diagram(bin_data, draw_ece=True, draw_mce=True, draw_bin_importance='alpha',
                            title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        plt.savefig(f"plots/{dataset}/reliability_diagrams/{optimizer_name}.pdf")
        plt.show()


def plot_results_wrt_parameters(runs, plot_values=['Accuracy', 'Top-5 Accuracy', 'ECE', 'MCE']):
    if type(runs) == dict:
        runs = [runs]
    results = collect_results(runs).copy().sort_values(by=['k', 'Structure'], ascending=[True, False])
    results.rename(columns={'Test Accuracy': 'Accuracy', 'Top-k Accuracy': 'Top-5 Accuracy'}, inplace=True)
    results[plot_values] = results[plot_values].apply(lambda x: x.str.split(' ').str[0]).astype(float)
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
        if not os.path.exists(f"plots/{dataset}/results_parameters"):
            os.makedirs(f"plots/{dataset}/results_parameters", exist_ok=True)
        for parameter in ['k', 'M', 'gamma']:
            for metric in plot_values:
                fig, ax = plt.subplots()
                title = rf"{metric} w.r.t. {parameter}"
                plot = sns.catplot(data=results[results.Dataset == dataset.upper()], x=parameter, y=metric,
                            hue='Structure', errorbar='sd', kind='box', legend=False, ax=ax)
                plot.ax.set_title(title)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plot.fig.tight_layout()
                plt.savefig(f"plots/{dataset}/results_parameters/{parameter}_{metric}.pdf")
                plt.show()


def plot_corrupted_data(runs, plot_values=['Accuracy', 'Top-5 Accuracy', 'ECE', 'MCE']):
    if type(runs) == dict:
        runs = [runs]

    runs = [run for run in runs if run['dataset'].lower() in ['cifar10', 'cifar100']]
    if len(runs) == 0:
        return
    # Plot corrupted reliability diagrams for each dataset, model and optimizer
    # plot_corrupted_reliability_diagrams(runs)

    # Plot corruption errors per dataset and optimizer and grouped by model and error
    plot_corrupted_results(runs, plot_values)

    # Plot robustness of all methods for each dataset, model and corruption type
    # plot_robustness(runs)


def plot_corrupted_reliability_diagrams(runs: Union[List[dict], dict]) -> None:
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        dataset = run['dataset']
        if not dataset.lower() in ['cifar10', 'cifar100']:
            continue
        if not os.path.exists(f"plots/{dataset}/corrupted/reliability_diagrams"):
            os.makedirs(f"plots/{dataset}/corrupted/reliability_diagrams", exist_ok=True)
        model = run['model_name']
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
            optimizer_name = f"NGD ({structure}, k = {params['k']}, " \
                             f"M = {params['mc_samples']}, gamma = {params['gamma']})"
        else:
            optimizer = run['optimizer_name']
            optimizer_name = run['optimizer_name']
        corrupted_results = run['corrupted_results']
        corrupted_bin_data = corrupted_results['bin_data']
        for severity in SEVERITY_LEVELS:
            bin_data_list = [
                corrupted_bin_data[(severity, corruption_type)]
                for corruption_type in ['noise', 'weather', 'blur', 'digital']
            ]
            corrupted_bin_data[(severity, 'all')] = merge_bin_data(bin_data_list)

        # Plot corrupted data for each severity level with severity levels along the x-axis
        # and corruption type along the y-axis
        for severity, corruption_type in corrupted_bin_data.keys():
            if corruption_type == 'clean':
                continue
            plt.figure()
            title = f"Reliability Diagram on {dataset.upper()}--C using {model}\n" \
                    f"(s = {severity}, c = {corruption_type.title()})\n" \
                    f"{optimizer}"
            reliability_diagram(corrupted_bin_data[(severity, corruption_type)], draw_ece=True, draw_mce=True,
                                draw_bin_importance='alpha', title=title, draw_averages=True, figsize=(6, 6), dpi=100)
            plt.savefig(f"plots/{dataset}/corrupted/reliability_diagrams/"
                        f"{optimizer_name}_{severity}_{corruption_type}.pdf")
            plt.show()
        for corruption_type in CORRUPTION_TYPES.keys():
            title = f"Reliability Diagram on {dataset.upper()}--C using {model}\n" \
                    f"(c = {corruption_type.title()})\n" \
                    f"{optimizer}"
            reliability_diagram(
                merge_bin_data([corrupted_bin_data[(severity, corruption_type)] for severity in SEVERITY_LEVELS]),
                draw_ece=True, draw_mce=True, draw_bin_importance='alpha', title=title,
                draw_averages=True, figsize=(6, 6), dpi=100)
            plt.savefig(f"plots/{dataset}/corrupted/reliability_diagrams/"
                        f"{optimizer_name}_{corruption_type}.pdf")
            plt.show()


def plot_corrupted_results(runs: Union[List[dict], dict],
                           plot_values=['Accuracy', 'Top-5 Accuracy', 'ECE', 'MCE'],
                           parameters=['k', 'M', 'gamma']) -> None:
    corrupted_results_df = collect_corrupted_results_df(runs)

    for dataset in corrupted_results_df['dataset'].unique():
        if not dataset.lower() in ['cifar10', 'cifar100']:
            continue
        if not os.path.exists(f"plots/{dataset}/corrupted/results"):
            os.makedirs(f"plots/{dataset}/corrupted/results", exist_ok=True)
        if not os.path.exists(f"plots/{dataset}/corrupted/results_parameters"):
            os.makedirs(f"plots/{dataset}/corrupted/results_parameters", exist_ok=True)
        plot_value = plot_values.copy()
        if dataset.lower() == 'cifar10':
            plot_value.remove('Top-5 Accuracy')
        # plot corruption errors per dataset and optimizer grouped by model and error (ECE, accuracy, etc.)
        # 2 x len(plot_values) grid of plots
        for model in set(corrupted_results_df[corrupted_results_df['dataset'] == dataset]['model']):
            sub_df = corrupted_results_df[(corrupted_results_df['dataset'] == dataset) &
                                          (corrupted_results_df['model'] == model)].copy().\
                sort_values(by=['k', 'structure'], ascending=[True, False]).drop_duplicates()
            sub_df[plot_value] *= 100
            for i, type in enumerate(CORRUPTION_TYPES.keys()):
                for j, value in enumerate(plot_value):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    if type == 'all':
                        plot = sns.boxplot(data=sub_df, x='severity', y=value, hue='method', ax=ax)
                        value_counts = sub_df.set_index(
                            ['severity', 'method']).sort_values(by=['severity', 'method'],
                                                                   ascending=[True, False]).\
                            groupby(['severity', 'method'], sort=False).apply(lambda x: x.value_counts().sum()).values
                    else:
                        plot = sns.boxplot(data=sub_df[sub_df['corruption_type'].isin(['clean', type])],
                                           x='severity', y=value, hue='method', ax=ax)
                        value_counts = sub_df[sub_df['corruption_type'].isin(['clean', type])].set_index(
                            ['severity', 'method']).sort_values(by=['severity', 'method'],
                                                                   ascending=[True, False]). \
                            groupby(['severity', 'method'], sort=False).apply(lambda x: x.value_counts().sum()).values

                    lines_per_boxplot = len(ax.lines) // len(ax.artists)
                    for i, (box, xtick, ytick) in enumerate(
                            zip(ax.artists, ax.get_xticklabels(), ax.get_yticklabels())
                    ):
                        color = box.get_facecolor()
                        line = ax.lines[i * lines_per_boxplot + 4]  # the median
                        if value_counts[i] == 1:
                            line.set_color(color)
                            w = line.get_linewidth()
                            line.set_linewidth(2 * w)
                    title = f"Corruption Errors on {dataset.upper()} for {model} on Corruption Type '{type.title()}'"
                    ax.set_title(title)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    plt.ylabel(f"{value} (\%)")
                    plt.tight_layout()
                    plt.savefig(f"plots/{dataset}/corrupted/results/{type}_{value}.pdf")
                    plt.show()
                    for parameter in parameters:
                        plt.figure()
                        if type == 'all':
                            plot = sns.catplot(data=sub_df, x=parameter, y=value, hue='structure', errorbar='sd', kind='box', legend=False)
                        else:
                            plot = sns.catplot(data=sub_df[sub_df['corruption_type'].isin(['clean', type])],
                                        x=parameter, y=value, hue='structure', errorbar='sd', kind='box', legend=False)
                        title = f"Corruption Errors on {dataset.upper()} for {model} on Corruption Type '{type.title()}'"
                        plot.ax.set_title(title)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                        plot.fig.tight_layout()
                        plt.savefig(f"plots/{dataset}/corrupted/results_parameters/{type}_{parameter}_{value}.pdf")
                        plt.show()


def plot_robustness(runs: Union[List[dict], dict]) -> None:
    df_mce = collect_corruption_errors(runs)
    df_rmce = collect_rel_corruption_errors(runs)
    datasets = list(set([run['dataset'] for run in runs]))
    parameters = ['k', 'M', 'gamma']
    for c in CORRUPTION_TYPES.keys():
        df_mce[c] *= 100
        df_mce[f"{c}_std"] *= 100
        df_rmce[c] *= 100
        df_rmce[f"{c}_std"] *= 100

    # Plot corruption errors and relative corruption errors as a function of optimizer accuracy
    for dataset in datasets:
        if not os.path.exists(f"plots/{dataset}/corrupted/robustness"):
            os.makedirs(f"plots/{dataset}/corrupted/robustness", exist_ok=True)
        if not os.path.exists(f"plots/{dataset}/corrupted/robustness_parameters/"):
            os.makedirs(f"plots/{dataset}/corrupted/robustness_parameters/", exist_ok=True)
        for model in df_mce[df_mce['dataset'] == dataset]['model_name'].unique():
            sub_df_mce = df_mce[
                (df_mce['dataset'] == dataset) &
                (df_mce['model_name'] == model)
                ].copy().sort_values(by='accuracy')
            sub_df_rmce = df_rmce[
                (df_rmce['dataset'] == dataset) &
                (df_rmce['model_name'] == model)
                ].copy().sort_values(by='accuracy')
            for type in CORRUPTION_TYPES.keys():
                accuracy = sub_df_mce['accuracy']
                mce_mean = sub_df_mce[type]
                rmce_mean = sub_df_rmce[type]
                mce_std = sub_df_mce[type + '_std']
                rmce_std = sub_df_rmce[type + '_std']
                fig, ax = plt.subplots()
                plt.plot(accuracy, mce_mean, label='mCE', marker='o', color='blue', linestyle='dashed', alpha=0.5)
                plt.fill_between(accuracy, y1=mce_mean-mce_std, y2=mce_mean+mce_std, alpha=0.1, color='blue')
                a, b = np.polyfit(accuracy, mce_mean, 1)
                plt.plot(accuracy, a * accuracy + b, color='blue')
                plt.plot(accuracy, rmce_mean, label='Relative mCE', marker='o', color='orange', linestyle='dashed', alpha=0.5)
                plt.fill_between(accuracy, y1=rmce_mean-rmce_std, y2=rmce_mean+rmce_std, alpha=0.1, color='orange')
                a, b = np.polyfit(accuracy, rmce_mean, 1)
                plt.plot(accuracy, a * accuracy + b, color='orange')
                ax.set_title(f"Robustness of Methods on {dataset.upper()} for {model} and Corruption Type '{type.title()}'")
                ax.set_xlabel('Test Accuracy (\%)')
                ax.set_ylabel('\%')
                ax.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join(f"{i+1}: {name}" for i, name in enumerate(sub_df_mce['optimizer_name']))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                # for i, (x, y) in enumerate(zip(accuracy, mce_mean)):
                #     ax.annotate(i+1, (x, np.min(mce_mean)),
                #                 xytext=(x, np.min(mce_mean)), ha='center', fontsize=16, transform=ax.transData)
                plt.savefig(f"plots/{dataset}/corrupted/robustness/{type}.pdf")
                plt.show()

                for parameter in parameters:
                    plt.figure()
                    sns.catplot(
                        data=sub_df_mce.rename(columns={type: 'mCE'}).sort_values(
                            by=['k', 'Structure'], ascending=[True, False]
                        ), x=parameter, y='mCE', hue='Structure', errorbar='sd', kind='box')
                    plt.savefig(f"plots/{dataset}/corrupted/robustness_parameters/mCE_{parameter}_{type}.pdf")
                    plt.show()
                    plt.figure()
                    sns.catplot(
                        data=sub_df_rmce.rename(columns={type: 'Rel. mCE'}).sort_values(
                            by=['k', 'Structure'], ascending=[True, False]
                        ), x=parameter, y='Rel. mCE', hue='Structure', errorbar='sd', kind='box')
                    plt.savefig(f"plots/{dataset}/corrupted/robustness_parameters/rmCE_{parameter}_{type}.pdf")
                    plt.show()


def make_csv(runs):
    if not os.path.exists('results'):
        os.mkdir('results')
    collect_results(runs).copy().to_csv('results/results.csv', index=False)
    collect_corrupted_results_df(runs).copy().to_csv('results/corrupted_results.csv', index=False)
    collect_corruption_errors(runs).copy().to_csv('results/corruption_errors.csv', index=False)
    collect_rel_corruption_errors(runs).copy().to_csv('results/rel_corruption_errors.csv', index=False)
    # results_table().copy().to_csv('results/table.csv', index=True)


def results_table():
    corrupted_results = pd.read_csv('results/corrupted_results.csv')
    corruption_errors = pd.read_csv('results/corruption_errors.csv')
    rel_corruption_errors = pd.read_csv('results/rel_corruption_errors.csv')
    corruption_errors.index = corruption_errors.optimizer_name
    rel_corruption_errors.index = rel_corruption_errors.optimizer_name

    f_interval = lambda x: rf"{100 * x.mean():.1f} ($\pm$ {100 * x.std():.1f})"
    corrupted_results = corrupted_results[corrupted_results.corruption_type != 'clean']. \
        drop(['severity', 'Loss', 'Top-5 Accuracy'], axis=1).groupby(['optimizer', 'corruption_type']).\
        agg(f_interval).unstack('corruption_type').swaplevel(axis=1)
    corrupted_results_all = pd.read_csv('results/corrupted_results.csv')

    corrupted_results_all = corrupted_results_all[corrupted_results_all.corruption_type != 'clean']. \
        drop(['severity', 'Loss', 'Top-5 Accuracy'], axis=1).groupby(['optimizer']).agg(f_interval)
    corrupted_results = pd.concat([corrupted_results, pd.concat({'all': corrupted_results_all}, axis=1)], axis=1)

    for c in CORRUPTION_TYPES.keys():
        f_interval = lambda x: rf"{100 * x[c]:.1f} ($\pm$ {100 * x[f'{c}_std']:.1f})"
        corrupted_results[c, 'mCE'] = corruption_errors[[c, f"{c}_std"]].apply(
            f_interval, axis=1
        )
        corrupted_results[c, 'Rel. mCE'] = rel_corruption_errors[[c, f"{c}_std"]].apply(
            f_interval, axis=1
        )
    corrupted_results.sort_index(inplace=True)
    corrupted_results.rename(
        columns={'all': 'All', 'noise': 'Noise', 'blur': 'Blur', 'weather': 'Weather', 'digital': 'Digital'}, inplace=True
    )
    corrupted_results = corrupted_results.reindex(
        columns=corrupted_results.columns.reindex(['All', 'Noise', 'Blur', 'Weather', 'Digital'], level=0)[0]
    )
    corrupted_results = corrupted_results.reindex(
        columns=corrupted_results.columns.reindex(['mCE', 'Rel. mCE', 'Accuracy', 'ECE', 'MCE'], level=1)[0]
    )
    return corrupted_results


def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels)
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]


def get_methods_and_model(dataset, model, model_params, optimizer, ngd_params=None, baseline='SGD'):
    optimizer = optimizer.__name__
    if model == 'DeepEnsemble':
        runs = load_all_runs()
        runs = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
                and run['dataset'].lower() == dataset.lower()]
        assert(len(runs) > 0)
        models = []
        for run in runs:
            state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt")['model_state_dict']
            model = Model(run['model_name'], **model_params)
            model.load_state_dict(state_dict=state_dict)
            models.append(model)
        model = DeepEnsemble(models=models, **model_params)
        methods = ['Deep Ensemble']
    elif model == 'HyperDeepEnsemble':
        runs = load_all_runs()
        runs = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
                and run['dataset'].lower() == dataset.lower()]
        assert(len(runs) > 0)
        models = []
        optimizers = []
        for run in runs:
            state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt")

            model = Model(run['model_name'], **model_params)
            model.load_state_dict(state_dict=state_dict['model_state_dict'])
            models.append(model)

            optimizer = StructuredNGD(model.parameters(), 50000)
            optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
            optimizers.append(optimizer)
        model = HyperDeepEnsemble(models=models, optimizers=optimizers, **model_params)
        methods = ['Hyper Deep Ensemble']
    elif model == 'BBB':
        assert(optimizer == baseline)
        runs = load_all_runs()
        run = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], bnn=True, **model_params)
        methods = ['BBB']
    elif model == 'Dropout':
        assert(optimizer == baseline)
        runs = load_all_runs()
        run = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], dropout_layers='all', p=0.2, **model_params)
        methods = ['Dropout']
    elif model == 'LLDropout':
        assert(optimizer == baseline)
        runs = load_all_runs()
        run = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], dropout_layers='last', p=0.2, **model_params)
        methods = ['LL Dropout']
    elif model == 'TempScaling':
        assert(optimizer == baseline)
        runs = load_all_runs()
        run = [run for run in runs if run['optimizer_name'] == baseline and run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt")['model_state_dict']
        model = Model(run['model_name'], **model_params)
        model.load_state_dict(state_dict=state_dict)
        model = TempScaling(model)
        methods = ['Temp Scaling']
    else:
        model = Model(model_type=model, **model_params)
        if optimizer == baseline:
            methods = ['Vanilla']
        elif optimizer.startswith('StructuredNGD'):
            methods = []
            for param in ngd_params:
                structure = param['structure'].replace('_', ' ').title().replace(' ', '')
                method = rf"NGD (structure = {structure}, $k = {param['k']})$"
                methods.append(method)
    return methods, model


class EarlyStopping:
    def __init__(self,
                 min_delta=0,
                 mode='min',
                 patience=10):
        self.min_delta = min_delta
        self.mode = 1 if mode == 'min' else -1
        self.patience = patience
        self.wait = 0
        self.best_val = self.mode * 1e15
        self.stopped_epoch = 0
        self.stop_training = False

    def on_train_begin(self):
        self.wait = 0
        self.best_val = self.mode * 1e15
        self.stop_training = False

    def on_epoch_end(self, epoch, current_val):
        if current_val is None:
            pass
        else:
            if self.mode * (current_val - self.best_val) < -self.mode * self.min_delta:
                self.best_val = current_val
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.stop_training = True
                self.wait += 1

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print(f"\nTerminated Training for Early Stopping at Epoch {self.stopped_epoch}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    plt.rcParams.update({'axes.titlesize': 16, 'text.usetex': True, 'figure.figsize': (12, 8)})

    runs = load_all_runs()
    plot_runs(runs)
