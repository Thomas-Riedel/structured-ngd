import argparse
import os.path
import sys
import datetime
import re
import itertools
from typing import Any

import torch.nn as nn
import torch.utils.data
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CIFAR100, STL10, SVHN
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from optimizers.rank_k_cov import *
from torch.optim import *
from reliability_diagrams import reliability_diagram
from corruption import *
from metrics import *

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
    parser = argparse.ArgumentParser(description='Run noisy optimizers with parameters.')
    parser.add_argument('-o', '--optimizers', type=str, default='StructuredNGD',
                        help='Optimizers, one of Adam, SGD, StructuredNGD (capitalization matters!, default: StructuredNGD)')
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
    args.optimizers = list(map(lambda x: eval(x), args.optimizers.split(',')))

    args_dict = dict(
        optimizers=args.optimizers,
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

    if dataset.lower() in ['imagenet', 'stl10']:
        img_size = (96, 96)
    else:
        img_size = (32, 32)

    if dataset.lower().startswith('cifar'):
        transform_augmented = CIFAR_TRANSFORM_AUGMENTED
        transform = CIFAR_TRANSFORM
    else:
        transform_augmented = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_size, padding=pad_size),
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
        training_data = MNIST('data/mnist/train', download=True, train=True, transform=transform_augmented)
        test_data = MNIST('data/mnist/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "fmnist":
        training_data = FashionMNIST('data/fmnist/train', download=True, train=True, transform=transform_augmented)
        test_data = FashionMNIST('data/fmnist/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "cifar10":
        training_data = CIFAR10('data/cifar10/train', download=True, train=True, transform=transform_augmented)
        test_data = CIFAR10('data/cifar10/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "cifar100":
        training_data = CIFAR100('data/cifar100/train', download=True, train=True, transform=transform_augmented)
        test_data = CIFAR100('data/cifar100/test', download=True, train=False, transform=transform)
    elif dataset.lower() == "stl10":
        training_data = STL10('data/stl10/train', download=True, split='train', transform=transform_augmented)
        test_data = STL10('data/stl10/test', download=True, split='test', transform=transform)
    elif dataset.lower() == "imagenet":
        training_data = ImageNet('data/imagenet/train', download=True, split='train', transform=transform_augmented)
        test_data = ImageNet('data/imagenet/test', download=True, split='val', transform=transform)
    elif dataset.lower() == "svhn":
        training_data = SVHN('data/svhn/train', download=True, split='train', transform=transform_augmented)
        test_data = SVHN('data/svhn/test', download=True, split='test', transform=transform)
    else:
        raise ValueError(f"Dataset {dataset} not recognized! Choose one of "
                         f"[mnist, fmnist, cifar10, cifar100, stl10, imagenet, svhn]")

    indices = list(range(len(training_data)))
    np.random.shuffle(indices)
    split = int(split * len(training_data))
    train_sampler = SubsetRandomSampler(indices[:split])
    val_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(
        training_data, sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        training_data, sampler=val_sampler, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True)

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


def run(epochs: int, model: nn.Module, optimizers: List[Union[Optimizer, StructuredNGD]],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, baseline: str = 'SGD',
        baseline_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [accuracy],
        eval_every: int = 100, n_bins: int = 10, mc_samples: int = 64) -> List[dict]:
    """Run a list of optimizers on data for multiple epochs using multiple hyperparameters and evaluate.

    :param epochs: int, number of epochs for training
    :param model: str, ResNet model for experiments
    :param optimizers: List[Union[baseline, StructuredNGD]], list of models to run experiments on
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
    loss_fn = nn.CrossEntropyLoss()
    runs = []
    dataset = train_loader.dataset.root.split('/')[1]

    for optim in optimizers:
        if optim is StructuredNGD:
            params = ngd_params
        else:
            params = baseline_params
        for param in params:
            print(optim.__name__, param)
            early_stopping = EarlyStopping()
            model.init_weights()
            if optim is StructuredNGD:
                optimizer = optim(model.parameters(), len(train_loader.dataset), **param)
            else:
                optimizer = optim(model.parameters(), **param)
            run = load_run(dataset, model, baseline)
            optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
            if (dataset.lower() in ['cifar10', 'cifar100']) and (optimizer_name != baseline) and (run is None):
                raise RuntimeError(f"Baseline {baseline} does not exist for this dataset and model!"
                                   f"Please first run the script with this baseline for the dataset and model.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

                scheduler.step()
                early_stopping.on_epoch_end(epoch, val_loss)
                if early_stopping.stop_training:
                    break

            test_loss, test_metrics, bin_data = model.evaluate(test_loader, metrics=metrics, optimizer=optimizer,
                                                               mc_samples=mc_samples, n_bins=n_bins)
            clean_results = dict(
                test_loss=test_loss,
                test_metrics=test_metrics,
                bin_data=bin_data
            )
            corrupted_results = get_corrupted_results(dataset, model, optimizer, baseline, metrics,
                                                      clean_results, mc_samples, n_bins)

            num_epochs = epoch + 1
            epoch_times = np.cumsum(epoch_times)
            total_time = epoch_times[-1]
            avg_time_per_epoch = epoch_times[-1] / num_epochs
            optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
            model_name = model.__name__
            num_params = model.num_params
            model_summary = model.summary

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            run = dict(
                optimizer_name=optimizer_name,
                model_name=model_name,
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
            torch.save(model, f"models/{timestamp}")
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
    runs = [run for run in runs if (run.get('dataset').lower() == 'cifar100') and (run['model_name'] == 'ResNet32')]
    # make_csv()

    # plot_loss(runs)
    # plot_metrics(runs)
    plot_generalization_gap(runs)
    # plot_reliability_diagram(runs)
    # plot_corrupted_data(runs)


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


def plot_metrics(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('plots/metrics'):
        os.makedirs('plots/metrics', exist_ok=True)
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
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
        plt.savefig(f"plots/metrics/metrics_{dataset}.pdf")


def plot_generalization_gap(runs: Union[dict, List[dict]]) -> None:
    """Plot generalization gap for each run.

    :param runs:
    :return:
    """
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
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
        plt.savefig(f"plots/{run['timestamp']}_generalization_gap.pdf")


def plot_reliability_diagram(runs: Union[dict, List[dict]]) -> None:
    """Plot reliability diagram for each run and save in plots folder.
    The function 'reliability_diagram' is adapted from https://github.com/hollance/reliability-diagrams

    :param runs:
    """
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('plots/reliability_diagrams'):
        os.makedirs('plots/reliability_diagrams', exist_ok=True)
    for run in runs:
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
            optimizer_name = f"NGD ({structure}, k = {params['k']}, " \
                             f"M = {params['mc_samples']}, gamma = {params['gamma']})"
        else:
            optimizer = run['optimizer_name']
            optimizer_name = run['optimizer_name']
        dataset = run['dataset']
        model = run['model_name']
        title = f"Reliability Diagram on {dataset.upper()} using {model}\n{optimizer}"
        plt.figure()
        bin_data = run['bin_data']
        reliability_diagram(bin_data, draw_ece=True, draw_mce=True, draw_bin_importance='alpha',
                            title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        plt.savefig(f"plots/reliability_diagrams/"
                    f"{run['timestamp']}_reliability_diagram_{dataset}_{model}_{optimizer_name}.pdf")


def plot_corrupted_data(runs, plot_values=['Accuracy', 'Top-5 Accuracy', 'ECE', 'MCE']):
    if type(runs) == dict:
        runs = [runs]

    runs = [run for run in runs if run['dataset'].lower() in ['cifar10', 'cifar100']]
    if len(runs) == 0:
        return
    # Plot corrupted reliability diagrams for each dataset, model and optimizer
    plot_corrupted_reliability_diagrams(runs)

    # Plot corruption errors per dataset and optimizer and grouped by model and error
    plot_corrupted_results(runs, plot_values)

    # Plot robustness of all optimizers for each dataset, model and corruption type
    plot_robustness(runs)


def plot_corrupted_reliability_diagrams(runs: Union[List[dict], dict]) -> None:
    if type(runs) == dict:
        runs = [runs]
    if not os.path.exists('plots/corrupted/reliability_diagrams'):
        os.makedirs('plots/corrupted/reliability_diagrams', exist_ok=True)
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        dataset = run['dataset']
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
                    f"(s = {severity}, type = {corruption_type})\n" \
                    f"{optimizer}"
            reliability_diagram(corrupted_bin_data[(severity, corruption_type)], draw_ece=True, draw_mce=True,
                                draw_bin_importance='alpha', title=title, draw_averages=True, figsize=(6, 6), dpi=100)
            plt.savefig(f"plots/corrupted/reliability_diagrams/"
                        f"{run['timestamp']}_{dataset}_{model}_{optimizer_name}_{severity}_{corruption_type}.pdf")
            plt.show()


def plot_corrupted_results(runs: Union[List[dict], dict],
                           plot_values=['Accuracy', 'Top-5 Accuracy', 'ECE', 'MCE']) -> None:
    corrupted_results_df = collect_corrupted_results_df(runs)
    if not os.path.exists('plots/corrupted/results'):
        os.makedirs('plots/corrupted/results', exist_ok=True)

    for dataset in corrupted_results_df['dataset'].unique():
        plot_value = plot_values.copy()
        if dataset.lower() == 'cifar10':
            plot_value.remove('Top-5 Accuracy')
        # plot corruption errors per dataset and optimizer grouped by model and error (ECE, accuracy, etc.)
        # 2 x len(plot_values) grid of plots
        for model in corrupted_results_df[corrupted_results_df['dataset'] == dataset]['model'].unique():
            sub_df = corrupted_results_df[(corrupted_results_df['dataset'] == dataset) &
                                          (corrupted_results_df['model'] == model)].copy()
            sub_df[plot_value] *= 100
            for i, type in enumerate(CORRUPTION_TYPES.keys()):
                plt.figure()
                for j, value in enumerate(plot_value):
                    if type == 'all':
                        sns.barplot(data=sub_df, x='severity', y=value, hue='optimizer')
                    else:
                        sns.barplot(data=sub_df[sub_df['corruption_type'].isin(['clean', type])],
                                    x='severity', y=value, hue='optimizer')
                    plt.title(f"Corruption Errors on {dataset.upper()} for {model} on Corruption Type '{type.title()}'")
                    plt.savefig(f"plots/corrupted/results/{dataset}_{model}_{type}_{value}.pdf")
                    plt.show()


def plot_robustness(runs: Union[List[dict], dict]) -> None:
    if not os.path.exists('plots/corrupted/robustness'):
        os.makedirs('plots/corrupted/robustness', exist_ok=True)
    df_mce = collect_corruption_errors(runs)
    df_rmce = collect_rel_corruption_errors(runs)
    datasets = list(set([run['dataset'] for run in runs]))

    # Plot corruption errors and relative corruption errors as a function of optimizer accuracy
    for dataset in datasets:
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
                accuracy = 100 * sub_df_mce['accuracy']
                mce_mean = 100 * sub_df_mce[type]
                rmce_mean = 100 * sub_df_rmce[type]
                mce_std = 100 * sub_df_mce[type + '_std']
                rmce_std = 100 * sub_df_rmce[type + '_std']
                fig, ax = plt.subplots()
                plt.plot(accuracy, mce_mean, label='mCE', marker='o', color='blue', linestyle='dashed', alpha=0.5)
                plt.fill_between(accuracy, y1=mce_mean-mce_std, y2=mce_mean+mce_std, alpha=0.1, color='blue')
                a, b = np.polyfit(accuracy, mce_mean, 1)
                plt.plot(accuracy, a * accuracy + b, color='blue')
                plt.plot(accuracy, rmce_mean, label='Relative mCE', marker='o', color='orange', linestyle='dashed', alpha=0.5)
                plt.fill_between(accuracy, y1=rmce_mean-rmce_std, y2=rmce_mean+rmce_std, alpha=0.1, color='orange')
                a, b = np.polyfit(accuracy, rmce_mean, 1)
                plt.plot(accuracy, a * accuracy + b, color='orange')
                ax.set_title(f"Robustness of Optimizers on {dataset.upper()} for {model} and Corruption Type '{type.title()}'")
                ax.set_xlabel('Test Accuracy (\%)')
                ax.set_ylabel('\%')
                ax.legend()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = '\n'.join(f"{i+1}: {name}" for i, name in enumerate(sub_df_mce['optimizer_name']))
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                # for i, (x, y) in enumerate(zip(accuracy, mce_mean)):
                #     ax.annotate(i+1, (x, np.min(mce_mean)),
                #                 xytext=(x, np.min(mce_mean)), ha='center', fontsize=16, transform=ax.transData)
                plt.savefig(f"plots/corrupted/robustness/{dataset}_{model}_{type}.pdf")
                plt.show()


def make_csv(directory: str = 'runs', baseline='SGD'):
    runs = load_all_runs(directory)
    results = pd.DataFrame(columns=['Dataset', 'Model', 'Optimizer', 'Training Loss', 'Test Loss',
                                    'Test Accuracy', 'Top-k Accuracy', 'ECE', 'MCE', 'Time (h)'])
    for run in runs:
        dataset = run['dataset']
        model = run['model_name']
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer = f"NGD (structure = {structure}, $k = {params['k']}, " \
                        f"M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
        else:
            optimizer = run['optimizer_name']
        baseline_run = load_run(dataset, model, baseline, directory)
        train_loss = run['train_loss'][-1]
        val_loss = run['val_loss'][-1]
        test_loss = run['test_loss']
        test_accuracy = run['test_metrics']['accuracy']
        top_k_accuracy = run['test_metrics']['top_5_accuracy']
        ece = run['bin_data']['expected_calibration_error']
        mce = run['bin_data']['max_calibration_error']
        comp_time = run['epoch_times'][-1]
        if run['optimizer_name'] == baseline:
            train_loss = compare(train_loss)
            val_loss = compare(val_loss)
            test_loss = compare(test_loss)
            test_accuracy = compare(test_accuracy)
            top_k_accuracy = compare(top_k_accuracy)
            ece = compare(ece)
            mce = compare(mce)
            comp_time = compare(comp_time / 3600)
        else:
            train_loss = compare(train_loss, baseline_run['train_loss'][-1])
            val_loss = compare(val_loss, baseline_run['val_loss'][-1])
            test_loss = compare(test_loss, baseline_run['test_loss'])
            test_accuracy = compare(test_accuracy, baseline_run['test_metrics']['accuracy'])
            top_k_accuracy = compare(top_k_accuracy, baseline_run['test_metrics']['top_5_accuracy'])
            ece = compare(ece, baseline_run['bin_data']['expected_calibration_error'])
            mce = compare(mce, baseline_run['bin_data']['max_calibration_error'])
            comp_time = compare(comp_time / 3600, baseline_run['epoch_times'][-1] / 3600)

        result = pd.DataFrame([{
            'Dataset': dataset,
            'Model': model,
            'Optimizer': optimizer,
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy,
            'Top-k Accuracy': top_k_accuracy,
            'ECE': ece, 'MCE': mce,
            'Time (h)': comp_time
        }])
        results = pd.concat([results, result], ignore_index=True)
    results.sort_values(by=['Dataset', 'Model', 'Optimizer'], inplace=True)
    if not os.path.exists('results'):
        os.mkdir('results')
    results.to_csv('results/results.csv', index=False)
    collect_corrupted_results_df(runs).to_csv('results/corrupted_results.csv', index=False)
    collect_corruption_errors(runs).to_csv('results/corruption_errors.csv', index=False)
    collect_rel_corruption_errors(runs).to_csv('results/rel_corruption_errors.csv', index=False)
    return results


def compare(x, y=None, f='{:.3f}'):
    if y is None:
        return f.format(x)
    sign = ['+', '', ''][1 - np.sign(x - y).astype(int)]
    relative_diff = 100 * (x / y - 1)
    return f"{f.format(x)} ({sign}{f.format(relative_diff)}%)"


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
    datasets = list(set(run['dataset'] for run in runs))
    for dataset in datasets:
        plot_runs([run for run in runs if run['dataset'] == dataset])
