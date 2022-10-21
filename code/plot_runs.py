import argparse
import sys
import datetime
import pickle
import re
import itertools
from typing import Any

import torch.nn as nn
import torch.utils.data
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CIFAR100, STL10, SVHN
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import Compose

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
                        help='Optimizers, one of Adam, StructuredNGD (capitalization matters!, default: StructuredNGD)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train models on data (default: 1)')
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR10",
                        help='Dataset for training, one of CIFAR10, CIFAR100, MNIST, FashionMNIST (default: CIFAR10)')
    parser.add_argument('-m', '--model', type=str, default="ResNet20",
                        help='ResNet model (default: ResNet20)')
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
    parser.add_argument('--mc_samples_eval', type=int, default=64,
                        help='Number of MC samples during evaluation (default: 64)')

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
        mc_samples_eval=args.mc_samples_eval
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


def load_data(dataset: str, batch_size: int, split: float = 0.8) -> \
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
            transforms.RandomCrop(32, padding=4),
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
        training_data = STL10('data/imagenet/train', download=True, split='train', transform=transform_augmented)
        test_data = STL10('data/imagenet/test', download=True, split='val', transform=transform)
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


def get_params(args: dict, add_weight_decay=True, n=1) -> dict:
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
    adam = [dict(lr=lr, weight_decay=add_weight_decay * prior_precision / n)
            for lr, prior_precision in zip(set(args['lr']), set(args['prior_precision']))]
    params = dict(ngd=ngd, adam=adam)
    return params


def run(epochs: int, model: nn.Module, optimizers: List[Union[Adam, StructuredNGD]],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
        adam_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [accuracy],
        eval_every: int = 100, n_bins: int = 10, mc_samples: int = 64) -> List[dict]:
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
    dataset = train_loader.dataset.root.split('/')[1]

    for optim in optimizers:
        if optim is StructuredNGD:
            params = ngd_params
        else:
            params = adam_params
        for param in params:
            print(optim.__name__, param)
            model.init_weights()
            if optim is StructuredNGD:
                optimizer = optim(model.parameters(), len(train_loader.dataset), **param)
            else:
                optimizer = optim(model.parameters(), **param)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            # Initialize computation times, losses and metrics for recording
            loss, metric, _ = model.evaluate(train_loader, metrics, optimizer=optimizer, mc_samples=1)
            epoch_times = [0.0]
            train_loss = []
            train_metrics = {}
            train_loss, train_metrics, _ = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

            loss, metric, _ = model.evaluate(val_loader, metrics=metrics, optimizer=optimizer, mc_samples=1)
            val_loss = []
            val_metrics = {}
            val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

            for epoch in range(epochs):
                # Train for one epoch
                loss, metric, comp_time = model.train(train_loader, optimizer, epoch=epoch,
                                                      loss_fn=loss_fn, metrics=metrics,
                                                      eval_every=eval_every)
                # Record epoch times, loss and metrics for epoch
                epoch_times.append(comp_time)
                train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

                # Record loss and metrics for epoch
                loss, metric, _ = model.evaluate(val_loader, metrics=metrics,
                                                 optimizer=optimizer, mc_samples=1)
                val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

                scheduler.step()

            test_loss, test_metrics, bin_data = model.evaluate(test_loader, metrics=metrics, optimizer=optimizer,
                                                               mc_samples=mc_samples, n_bins=n_bins)
            clean_results = dict(
                test_loss=test_loss,
                test_metrics=test_metrics,
                bin_data=bin_data
            )
            corrupted_results = get_corrupted_results(dataset, model, optimizer, metrics,
                                                      clean_results, mc_samples, n_bins)

            epoch_times = np.cumsum(epoch_times)
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
    plot_loss(runs)
    plot_metrics(runs)
    plot_generalization_gap(runs)
    plot_reliability_diagram(runs)
    plot_corrupted_results(runs)


def plot_loss(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    # Plot training loss in terms of epochs and computation time
    plt.figure(figsize=(12, 8))
    for run in runs:
        label = run['optimizer_name']
        plt.subplot(2, 1, 1)
        plt.plot(run['train_loss'], label=label)

        # plt.ylim(bottom=0)
        plt.title('Training Loss w.r.t. Epochs')
        plt.xlabel('epochs')
        plt.yscale('log')

        plt.subplot(2, 1, 2)
        plt.plot(run['epoch_times'], run['train_loss'], label=label)

        plt.title('Training Loss w.r.t Time')
        plt.xlabel('time (s)')
        plt.legend()
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/train_losses.pdf')

    # Plot validation loss over epochs and computation time
    plt.figure(figsize=(12, 8))
    for run in runs:
        label = run['optimizer_name']
        plt.subplot(2, 1, 1)
        plt.plot(run['val_loss'], label=label)
        plt.title('Validation Loss')
        plt.xlabel('epochs')
        # plt.ylim(bottom=0)
        plt.yscale('log')

        plt.subplot(2, 1, 2)
        plt.plot(run['epoch_times'], run['val_loss'], label=label)
        plt.title('Validation Loss')
        plt.xlabel('time (s)')
        # plt.ylim(bottom=0)
        plt.yscale('log')
        plt.legend()

    plt.tight_layout()
    plt.savefig('plots/val_losses.pdf')


def plot_metrics(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        # Plot metrics for training validation set wrt epochs
        plt.figure(figsize=(12, 8))
        label = run['optimizer_name']
        plt.subplot(2, 1, 1)
        for key in run['val_metrics'].keys():
            plt.plot(run['train_metrics'][key],
                     label=f"Train Data ({key})")
            plt.plot(run['val_metrics'][key],
                     label=f"Validation Data ({key})")

        plt.title(f"Metrics w.r.t. Epochs ({label})")
        plt.xlabel('epochs')
        plt.ylim(0, 1)
        # plt.legend()
        # plt.savefig(f"plots/{run['timestamp']}_metrics.pdf")

        # Plot metrics for training validation set wrt computation time
        # plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 2)
        for key in run['val_metrics'].keys():
            plt.plot(run['epoch_times'], run['train_metrics'][key],
                     label=f"Train Data ({key})")
            plt.plot(run['epoch_times'], run['val_metrics'][key],
                     label=f"Validation Data ({key})")
        plt.title(f"Metrics w.r.t. Time ({label})")
        plt.xlabel('time (s)')
        plt.ylim(0, 1)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"plots/{run['timestamp']}_metrics.pdf")


def plot_generalization_gap(runs: Union[dict, List[dict]]) -> None:
    """Plot generalization gap for each run.

    :param runs:
    :return:
    """
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        label = run['optimizer_name']

        # Plot generalization gap in terms wrt epochs
        plt.figure(figsize=(12, 8))
        min_index = np.argmin(run['val_loss'])
        ymax = np.concatenate((run['train_loss'], run['val_loss'])).max()
        plt.plot(run['train_loss'], label='Train Data')
        plt.plot(run['val_loss'], label='Validation Data')
        plt.vlines(min_index, ymin=0, ymax=ymax, colors='r', linestyle='dashed')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title(f"Generalization Gap ({label})")

        # # Plot generalization gap in terms wrt time
        # plt.figure(figsize=(12, 8))
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
    for run in runs:
        label = run['optimizer_name']
        title = f"Reliability Diagram ({label})"
        plt.figure(figsize=(12, 8))
        bin_data = run['bin_data']
        reliability_diagram(bin_data, draw_ece=True, draw_mce=True,
                            title=title, draw_averages=True,
                            figsize=(6, 6), dpi=100)
        plt.savefig(f"plots/{run['timestamp']}_reliability_diagram.pdf")


def record_loss_and_metrics(losses, metrics, loss, metric):
    if len(metrics.keys()) == 0:
        for key in metric:
            metrics[key] = []
    assert (metrics.keys() == metric.keys())
    losses.append(loss)
    for key in metric.keys():
        metrics[key].append(metric[key])
    return losses, metrics


def plot_corrupted_results(runs, plot_values=['mce', 'ece', 'accuracy']):
    if type(runs) == dict:
        runs = [runs]

    # Plot corrupted reliability diagrams for each dataset, model and optimizer
    plot_corrupted_reliability_diagrams(runs)

    # Plot corruption errors per dataset and optimizer and grouped by model and error
    plot_corrupted_results(runs, plot_values)

    # Plot robustness of all optimizers for each dataset, model and corruption type
    plot_robustness(runs)


def plot_corrupted_reliability_diagrams(runs: Union[List[dict], dict]) -> None:
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        dataset = run['dataset']
        model = run['model_name']
        optimizer = run['optimizer_name']

        corrupted_results = run['corrupted_results']
        clean_bin_data = run['bin_data']
        corrupted_bin_data = corrupted_results['bin_data']
        label = corrupted_results['optimizer']

        plt.figure(figsize=(12, 8))
        plt.title(f"Reliability Diagram {label}")
        nrows = len(CORRUPTION_TYPES)
        ncols = 6

        # Plot clean bin data (repeatedly in first column)
        plt.figure(figsize=(12, 8))
        severity = 0
        corruption_type = 'clean'
        for i in range(ncols):
            plt.subplot(nrows, ncols, (severity + 1) + nrows * i)
            title = f"severity = {severity}, corruption = {corruption_type}"
            reliability_diagram(clean_bin_data, draw_ece=True, draw_mce=True,
                                title=title, draw_averages=True,
                                figsize=(6, 6), dpi=100)
        # Plot corrupted data for each severity level with severity levels along the x-axis
        # and corruption type along the y-axis
        for severity, corruption_type in corrupted_bin_data.keys():
            index = (severity + 1) + nrows * CORRUPTION_TYPES.index(corruption_type)
            plt.subplot(nrows, ncols, index)
            title = f"severity = {severity}, corruption = {corruption_type}"
            reliability_diagram(corrupted_bin_data[(severity, corruption_type)], draw_ece=True, draw_mce=True,
                                title=title, draw_averages=True,
                                figsize=(6, 6), dpi=100)
        corruption_types = ', '.join('clean', *CORRUPTION_TYPES.keys())
        plt.title(f"Reliability Diagrams ({dataset}; {model}; {optimizer})")
        # ; (→: severity (0-6); ↓: corruption_type: ({corruption_types}))")
        plt.tight_layout()
        plt.savefig(f"plots/{run['timestamp']}_reliability_diagram_{dataset}_{model}_{optimizer}.pdf")


def plot_corrupted_results(runs: Union[List[dict], dict], plot_values=['ece', 'mce', 'accuracy']) -> None:
    corrupted_results_df = collect_corrupted_results_df(runs)
    for dataset in corrupted_results_df['dataset'].unique():
        # plot corruption errors per dataset and optimizer grouped by model and error (ECE, accuracy, etc.)
        # 2 x len(plot_values) grid of plots
        for model in corrupted_results_df['dataset' == dataset]['model_name'].unique():
            fig, axes = plt.subplots(2, len(plot_values), sharex=True, squeeze=False)
            sub_df = corrupted_results_df[(corrupted_results_df['dataset'] == dataset) &
                                          (corrupted_results_df['model_name'] == model)].copy()
            sub_df[plot_values] *= 100
            corr_handles = []
            corr_labels = []
            opt_handles = []
            opt_labels = []
            for i, value in enumerate(plot_values):
                axes[0, i].set_title(value)
                corruption_plot = sns.barplot(ax=axes[0, i], data=sub_df, x='severity', y=value, hue='corruption_type')
                optimizer_plot = sns.barplot(ax=axes[1, i], data=sub_df, x='severity', y=value, hue='optimizer_name')
                axes[0, i].get_legend().remove()
                axes[1, i].get_legend().remove()
                corr_handles.append(corruption_plot)
                corr_labels.append(corruption_plot.get_label())
                opt_handles.append(optimizer_plot)
                opt_labels.append(optimizer_plot.get_label())
            axes[0, -1].legend(handles=corr_handles, labels=corr_labels, bbox_to_anchor=(1.3, 1.2))
            axes[1, -1].legend(handles=opt_handles, labels=opt_labels, bbox_to_anchor=(1.3, 1.2))
            fig.suptitle(f"Corruption Errors on {dataset.upper()} for {model}")
            plt.tight_layout()
            plt.savefig(f"plots/corrupted_results_{dataset}_{model}.pdf")
            plt.show()


def plot_robustness(runs: Union[List[dict], dict]) -> None:
    df_mce = collect_corruption_errors(runs)
    df_rmce = collect_rel_corruption_errors(runs)

    # Plot corruption errors and relative corruption errors as a function of optimizer accuracy
    for dataset in df_mce['dataset'].unique():
        for model in df_mce[df_mce['dataset'] == dataset]['model_name'].unique():
            sub_df_mce = df_mce[
                (df_mce['dataset'] == dataset) &
                (df_mce['model_name'] == model)
                ].copy()
            sub_df_rmce = df_rmce[
                (df_rmce['dataset'] == dataset) &
                (df_rmce['model_name'] == model)
                ].copy()
            for type in CORRUPTION_TYPES.keys():
                mce_mean = 100 * sub_df_mce[type]
                rmce_mean = 100 * sub_df_rmce[type]
                mce_std = (100 ** 2) * sub_df_mce[type + '_std']
                rmce_std = (100 ** 2) * sub_df_rmce[type + '_std']
                plt.figure(figsize=(12, 8))
                plt.plot(100 * sub_df_mce['accuracy'], mce_mean, label='mCE')
                plt.fill_between(100 * sub_df_mce['accuracy'], y1=mce_mean-mce_std, y2=mce_mean+mce_std)
                for label, x, y in zip(sub_df_mce['optimizer_name'], sub_df_mce['accuracy'], sub_df_mce[type]):
                    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, -10), ha='center')
                plt.plot(100 * sub_df_rmce['accuracy'], rmce_mean, label='Relative mCE')
                plt.fill_between(100 * sub_df_mce['accuracy'], y1=rmce_mean-rmce_std, y2=rmce_mean+rmce_std)
                plt.title(f"Robustness of Optimizers on {dataset} for {model} and corruption type {type}")
                plt.xlabel('Test Accuracy (%)')
                plt.ylabel('%')
                plt.legend()
                plt.savefig(f"plots/robustness_{dataset}_{model}_{type}.pdf")
                plt.show()


def make_csv(directory: str = 'runs'):
    runs = load_all_runs(directory)
    results = pd.DataFrame(columns=['Dataset', 'Model', 'Optimizer', 'Training Loss', 'Test Loss',
                                    'Test Accuracy', 'Top-k Accuracy', 'ECE', 'MCE', 'Time'])
    for run in runs:
        dataset = run['dataset']
        model = run['model_name']
        optimizer = run['optimizer_name']

        adam_run = load_run(dataset, model, 'Adam', directory)
        train_loss = run['train_loss'][-1]
        test_loss = run['test_loss']
        test_accuracy = run['test_metrics']['accuracy']
        top_k_accuracy = run['test_metrics']['top_k_accuracy']
        ece = run['bin_data']['expected_calibration_error']
        mce = run['bin_data']['max_calibration_error']
        comp_time = run['epoch_times'][-1]
        if run['optimizer_name'] == 'Adam':
            train_loss = compare(train_loss)
            test_loss = compare(test_loss)
            test_accuracy = compare(test_accuracy)
            top_k_accuracy = compare(top_k_accuracy)
            ece = compare(ece)
            mce = compare(mce)
            comp_time = compare(comp_time / 3600)
        else:
            train_loss = compare(train_loss, adam_run['train_loss'][-1])
            test_loss = compare(test_loss, adam_run['test_loss'])
            test_accuracy = compare(test_accuracy, adam_run['test_metrics']['accuracy'])
            top_k_accuracy = compare(top_k_accuracy, adam_run['test_metrics']['top_k_accuracy'])
            ece = compare(ece, adam_run['bin_data']['expected_calibration_error'])
            mce = compare(mce, adam_run['bin_data']['max_calibration_error'])
            comp_time = compare(comp_time / 3600, adam_run['epoch_times'][-1])

        result = pd.DataFrame([{
            'Dataset': dataset,
            'Model': model,
            'Optimizer': optimizer,
            'Training Loss': train_loss,
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy,
            'Top-k Accuracy': top_k_accuracy,
            'ECE': ece, 'MCE': mce,
            'Time (h)': comp_time
        }])
        results = pd.concat([results, result], ignore_index=True)
    results.sort_values(by=['Dataset', 'Model', 'Optimizer'], inplace=True)
    return results


def compare(x, y=None, f='{:.3f}'):
    if y is None:
        return f.format(x)
    sign = ['+', '', ''][1 - np.sign(x - y).astype(int)]
    relative_diff = 100 * (x / y - 1)
    return f"{f.format(x)} ({sign}{f.format(relative_diff)}%)"


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    runs = load_all_runs()
    plot_runs(runs)
