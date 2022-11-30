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
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import colorcet as cc

from optimizers.rank_k_cov import *
from torch.optim import *
from reliability_diagrams import reliability_diagram, uncertainty_diagram
from corruption import *
from PIL import Image
from create_c import *
from metrics import *
from network import *
from util import *


def run_experiments(
        epochs: int, methods: List[str], model: nn.Module, optimizer: List[Union[Optimizer, StructuredNGD]],
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, baseline: str = 'SGD',
        baseline_params: List[dict] = None, ngd_params: List[dict] = None,  metrics: List[Callable] = [],
        eval_every: int = 100, n_bins: int = 10, mc_samples_test: int = 32, mc_samples_val: int = 1,
        batch_size: int = 128
) -> List[dict]:
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
        return evaluate(
            methods[0], model, train_loader, val_loader, test_loader,
            baseline, metrics, n_bins, mc_samples_val, mc_samples_test, batch_size
        )
    else:
        return train_and_evaluate(
            epochs, methods, model, optimizer, train_loader, val_loader, test_loader, baseline, baseline_params,
            ngd_params, metrics, eval_every, n_bins, mc_samples_val, mc_samples_test, batch_size
        )


def evaluate(
        method: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
        baseline: str = 'SGD', metrics: List[Callable] = [], n_bins: int = 10,
        mc_samples_val: int = 1, mc_samples_test: int = 32, batch_size: int = 128
):
    if isinstance(model, TempScaling):
        model.set_temperature(val_loader)
    runs = []
    dataset = train_loader.dataset.dataset.root.split('/')[1]

    train_loss = []
    train_metrics = {}
    loss, metric, _, _ = model.evaluate(train_loader, metrics=metrics, n_bins=n_bins)
    train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

    val_loss = []
    val_metrics = {}
    loss, metric, _, _ = model.evaluate(val_loader, metrics=metrics, mc_samples=mc_samples_val, n_bins=n_bins)
    val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

    test_loss, test_metrics, bin_data, uncertainty = model.evaluate(test_loader, metrics=metrics,
                                                                    mc_samples=mc_samples_test, n_bins=n_bins)
    clean_results = dict(
        test_loss=test_loss,
        test_metrics=test_metrics,
        bin_data=bin_data,
        uncertainty=uncertainty
    )
    corrupted_results = get_corrupted_results(dataset, model, baseline, method, baseline, metrics,
                                              clean_results, mc_samples_test, n_bins, batch_size)

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
        uncertainty=uncertainty,
        corrupted_results=corrupted_results,
        timestamp=timestamp
    )
    runs.append(run)
    save_runs(run)
    return runs


def train_and_evaluate(epochs: int, methods: List[str], model: nn.Module, optim: Union[Optimizer, StructuredNGD],
                       train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, baseline: str = 'SGD',
                       baseline_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [],
                       eval_every: int = 100, n_bins: int = 10, mc_samples_val: int = 1, mc_samples_test: int = 32,
                       batch_size: int = 128):
    loss_fn = nn.NLLLoss()
    runs = []
    dataset = train_loader.dataset.dataset.root.split('/')[1]
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
        run = load_run(dataset, model, baseline, method='Vanilla')
        optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
        if methods[i] != 'Vanilla' and run is None:
            raise RuntimeError(f"Baseline {baseline} does not exist for this dataset and model! "
                               f"Please first run the script with this baseline for the dataset and model.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8)

        # Initialize computation times, losses and metrics for recording
        # loss, metric, _, _ = model.evaluate(train_loader, metrics=metrics, optimizer=optimizer)
        epoch_times = [0.0]
        train_loss = []
        train_metrics = {}
        # train_loss, train_metrics = record_loss_and_metrics(train_loss, train_metrics, loss, metric)

        # loss, metric, _, _ = model.evaluate(val_loader, metrics=metrics,
        #                                     optimizer=optimizer, mc_samples=mc_samples_val)
        val_loss = []
        val_metrics = {}
        # val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

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
            loss, metric, _, _ = model.evaluate(val_loader, metrics=metrics,
                                                optimizer=optimizer, mc_samples=mc_samples_val)
            val_loss, val_metrics = record_loss_and_metrics(val_loss, val_metrics, loss, metric)

            scheduler.step(loss)
            early_stopping.on_epoch_end(epoch, loss)
            if early_stopping.stop_training:
                break
        early_stopping.on_train_end()

        test_loss, test_metrics, bin_data, uncertainty = model.evaluate(
            test_loader, metrics=metrics, optimizer=optimizer, mc_samples=mc_samples_test, n_bins=n_bins
        )
        clean_results = dict(
            test_loss=test_loss,
            test_metrics=test_metrics,
            bin_data=bin_data,
            uncertainty=uncertainty
        )
        corrupted_results = get_corrupted_results(dataset, model, optimizer, methods[i], baseline, metrics,
                                                  clean_results, mc_samples_test, n_bins, batch_size)

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
            uncertainty=uncertainty,
            corrupted_results=corrupted_results,
            timestamp=timestamp
        )
        runs.append(run)
        save_runs(run)
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_size': len(train_loader.dataset)
        }, f"checkpoints/{timestamp}.pt")
        print('Finished Training')
    return runs


def plot_runs(runs: Union[dict, List[dict]]) -> None:
    """Plot runs and save in plots folder.

    :param runs: List[dict], list of runs to be plotted
    """
    if type(runs) == dict:
        runs = [runs]
    os.makedirs('plots', exist_ok=True)
    make_csv(runs)

    # plot_results_wrt_parameters(runs)
    # plot_loss(runs)
    # plot_metrics(runs)
    # plot_generalization_gap(runs)
    plot_reliability_diagram(runs)
    # plot_ablation_study(runs)
    # plot_corrupted_data(runs)


def plot_loss(runs: Union[dict, List[dict]]) -> None:
    if type(runs) == dict:
        runs = [runs]
    # Plot training and validation loss in terms of epochs and computation time
    plt.figure()
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
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

            plt.title(f"Metrics w.r.t. Epochs ({dataset})")
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
            plt.title(f"Metrics w.r.t. Time ({dataset})")
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
    runs = load_all_runs(method_unique=True)
    for run in runs:
        dataset = run['dataset']
        os.makedirs(f"plots/{dataset}/reliability_diagrams", exist_ok=True)
        os.makedirs(f"plots/{dataset}/uncertainty_diagrams", exist_ok=True)
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            if structure in ['RankCov', 'Arrowhead'] and params['k'] == 0:
                structure = 'Diagonal'
            optimizer = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
            optimizer_name = f"NGD ({structure}, k = {params['k']}, " \
                             f"M = {params['mc_samples']}, gamma = {params['gamma']})"
            method = rf"{run['method'].split(' (')[0]} ({structure}, $k = {params['k']}$)"
        else:
            optimizer = run['optimizer_name']
            optimizer_name = run['optimizer_name']
            method = run['method'].replace(', $M = 1$', '')
        model = run['model_name']
        title = f"Reliability Diagram on {dataset}\n{method}"
        plt.figure()
        bin_data = run['bin_data']
        reliability_diagram(bin_data, draw_ece=True, draw_mce=True, draw_bin_importance='alpha',
                            title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        plt.savefig(f"plots/{dataset}/reliability_diagrams/{method}_{optimizer_name}.pdf")
        plt.show()

        title = f"Uncertainty Diagram on {dataset}\n{method}"
        plt.figure()
        uncertainty_diagram(bin_data, draw_uce=True, draw_muce=True, draw_bin_importance='alpha',
                            title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        plt.savefig(f"plots/{dataset}/uncertainty_diagrams/{method}_{optimizer_name}.pdf")
        plt.show()


def plot_ablation_study(runs, plot_values=['NLL', 'Accuracy', 'ECE',
                                           'UCE', 'MCE', 'MUCE', 'Top-5 Accuracy', 'SCE', 'ACE', 'BS',
                                           'MI', 'PU']):
    if type(runs) == dict:
        runs = [runs]
    runs = load_all_runs(method_unique=False)
    runs = [run for run in runs if run['dataset'] == 'CIFAR10']
    corr_errors = collect_corruption_errors(runs).copy()
    results = collect_results(runs).copy().sort_values(by=['k', 'Structure', 'Method'], ascending=[True, False, False])
    results['Structure'] = results['Structure'].apply(lambda x: 'RankCov' if x == 'Rankcov' else x)
    results.rename(columns={'Test Loss': 'NLL', 'Test Accuracy': 'Accuracy', 'Top-k Accuracy': 'Top-5 Accuracy'}, inplace=True)
    results[plot_values] = results[plot_values].apply(lambda x: x.str.split(' ').str[0]).astype(float)
    results = results.drop_duplicates(subset=['Method', 'k', 'M', 'gamma', 'Structure'], keep='first')
    plot_values += ['mCE', 'Rel. mCE']
    datasets = list(set([run['dataset'] for run in runs]))
    for dataset in datasets:
        os.makedirs(f"plots/{dataset}/results_parameters", exist_ok=True)
        for parameter in ['k', 'M', 'gamma']:
            for metric in plot_values:
                fig, ax = plt.subplots()
                title = rf"{metric} w.r.t. {parameter}"
                data = results[results.Dataset == dataset].copy()
                if parameter == 'M':
                    data = data[(data.gamma == 1) & (data.Method.str.startswith('NGD')) & (data.Structure == 'RankCov')]
                elif parameter == 'gamma':
                    data = data[(data.M == 1) & (data.Method.str.startswith('NGD')) & (data.Structure == 'RankCov')]
                else:
                    data = data[(data.k != '--') & (data.gamma == 1.0) & (data.M == 1)]
                data = data.sort_values(by=parameter, ascending=True)
                if metric in ['mCE', 'Rel. mCE']:
                    corr_error = corr_errors.copy().melt(id_vars=['Method', parameter, 'Structure', 'Accuracy'], value_vars=['mCE', 'Rel. mCE'])
                    corr_error = corr_error.groupby(['Method', parameter, 'Structure', 'variable']).mean().reset_index()
                    corr_error.drop(columns='Accuracy', axis=1, inplace=True)
                    corr_error = corr_error.pivot(index=['Method', parameter, 'Structure'], columns='variable', values='value')
                    data = data.merge(corr_error, on=['Method', parameter])
                if parameter == 'k':
                    plot = sns.scatterplot(data=data, x=parameter, y=metric, hue='Structure', legend=False, ax=ax)
                    plot = sns.lineplot(data=data, x=parameter, y=metric, hue='Structure', legend=True, ax=ax)
                else:
                    if parameter == 'gamma':
                        data['Method'] = data['Method'].apply(lambda x: x.replace('M = ', r'\gamma = '))
                    data = data.sort_values(by=['Method', 'Structure'])
                    plot = sns.scatterplot(data=data, x=parameter, y=metric, hue='Method', legend=False, ax=ax)
                    plot = sns.lineplot(data=data, x=parameter, y=metric, hue='Method', legend=True, ax=ax)
                ax.set_title(title, fontsize=24)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
                plt.tight_layout()
                # MCE and mCE are overwritten when saving
                if metric == 'mCE':
                    metric = 'meanCE'
                if parameter == 'gamma':
                    plt.xscale('log')
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.savefig(f"plots/{dataset}/results_parameters/{parameter}_{metric}_legend.pdf")
                plt.show()


def plot_corrupted_data(runs, plot_values=('NLL', 'Accuracy', 'ECE',
                                           'UCE', 'MCE', 'MUCE', 'Top-5 Accuracy', 'SCE', 'ACE', 'BS',
                                           'Model Uncertainty', 'Predictive Uncertainty')):
    if type(runs) == dict:
        runs = [runs]

    if len(runs) == 0:
        return
    runs = load_all_runs(method_unique=True)

    # show_corrupted_images(path='Tiny-ImageNet/val_1230.JPEG', corruption='gaussian_noise')
    # show_corrupted_images(path='ImageNet/ILSVRC2012_val_00000023.JPEG', corruption='gaussian_noise')

    # Plot corrupted reliability diagrams for each dataset, model and optimizer
    plot_corrupted_reliability_diagrams(runs)

    # Plot corruption errors per dataset and optimizer and grouped by model and error
    # plot_corrupted_results(runs, plot_values)

    # plot_shift_intensity_diagram(runs, plot_values)

    # Plot robustness of all methods for each dataset, model and corruption type
    # plot_robustness(runs)

    # Plot predictive uncertainty under data shift
    # plot_uncertainty(runs)


def show_corrupted_images(path='Tiny-ImageNet/val_1230.JPEG', corruption='gaussian_noise', show=False):
    dataset = os.path.dirname(path) + '-C'
    basename = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path)
    if np.array(img).shape != (64, 64, 3):
        img.thumbnail((224, 224))
    os.makedirs(f"plots/{dataset}", exist_ok=True)
    y = []
    for s in [0, 1, 2, 3, 4, 5]:
        np.random.seed(s)
        if s == 0:
            f = lambda x: x
        else:
            f = lambda x: eval(corruption)(x, s).astype(np.uint8)
        x = np.array(f(img))
        y.append(x)
        x_img = Image.fromarray(x)
        x_img.save(os.path.join('plots', dataset, f"{corruption}_{s}_{basename}.jpg"))
        if show:
            x_img.show()
    y = np.concatenate(y, axis=1)
    img = Image.fromarray(y)
    img.save(os.path.join('plots', dataset, f"{corruption}_{basename}.jpg"))
    if show:
        img.show()


def plot_corrupted_reliability_diagrams(runs: Union[List[dict], dict]) -> None:
    if type(runs) == dict:
        runs = [runs]
    for run in runs:
        dataset = run['dataset']
        os.makedirs(f"plots/{dataset}/corrupted/reliability_diagrams", exist_ok=True)
        os.makedirs(f"plots/{dataset}/corrupted/uncertainty_diagrams", exist_ok=True)
        method = run['method']
        # if run['optimizer_name'].startswith('StructuredNGD'):
        #     params = run['params']
        #     structure = params['structure'].replace('_', ' ').title().replace(' ', '')
        #     optimizer = rf"NGD ({structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
        #     optimizer_name = f"NGD ({structure}, k = {params['k']}, " \
        #                      f"M = {params['mc_samples']}, gamma = {params['gamma']})"
        corrupted_results = run['corrupted_results']
        corrupted_bin_data = corrupted_results['bin_data']
        corrupted_uncertainty = corrupted_results['uncertainty']
        for severity in SEVERITY_LEVELS:
            bin_data_list = [
                corrupted_bin_data[(severity, corruption_type)]
                for corruption_type in ['noise', 'weather', 'blur', 'digital']
            ]
            corrupted_bin_data[(severity, 'all')] = merge_bin_data(bin_data_list)
            corrupted_uncertainty['model_uncertainty'][(severity, 'all')] = np.concatenate([
                corrupted_uncertainty['model_uncertainty'][(severity, corruption_type)]
                for corruption_type in ['noise', 'weather', 'blur', 'digital']
            ])

        # Plot corrupted data for each severity level with severity levels along the x-axis
        # and corruption type along the y-axis
        # for severity, corruption_type in corrupted_bin_data.keys():
        #     if corruption_type == 'clean':
        #         continue
        #
        #     # Draw reliability diagram for severity and corruption types
        #     plt.figure()
        #     title = f"Reliability Diagram on {dataset}--C for Method {method}\n" \
        #             f"(s = {severity}, c = {corruption_type.title()})\n" \
        #             f"{method}"
        #     reliability_diagram(corrupted_bin_data[(severity, corruption_type)], draw_ece=True, draw_mce=True,
        #                         draw_bin_importance='alpha', title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        #     plt.savefig(f"plots/{dataset}/corrupted/reliability_diagrams/"
        #                 f"{method}_{severity}_{corruption_type}.pdf")
        #     plt.show()
        #
        #     # Draw uncertainty diagram for severity and corruption types
        #     plt.figure()
        #     title = f"Uncertainty Diagram on {dataset}--C for Method {method}\n" \
        #             f"(s = {severity}, c = {corruption_type.title()})\n" \
        #             f"{method}"
        #     uncertainty_diagram(corrupted_bin_data[(severity, corruption_type)], draw_uce=True, draw_muce=True,
        #                         draw_bin_importance='alpha', title=title, draw_averages=True, figsize=(6, 6), dpi=100)
        #     plt.savefig(f"plots/{dataset}/corrupted/uncertainty_diagrams/"
        #                 f"{method}_{severity}_{corruption_type}.pdf")
        #     plt.show()

        method = method.replace(', $M = 1$', '')
        for corruption_type in CORRUPTION_TYPES.keys():
            # Plot reliability diagram for corruption type
            plt.figure()
            if corruption_type == 'all':
                title = f"Reliability Diagram on {dataset}--C\n" \
                        f"{method}"
            else:
                title = f"Reliability Diagram on {dataset}--C\n" \
                        f"(c = {corruption_type.title()})\n" \
                        f"{method}"
            reliability_diagram(
                merge_bin_data([corrupted_bin_data[(severity, corruption_type)] for severity in SEVERITY_LEVELS]),
                draw_ece=True, draw_mce=True, draw_bin_importance='alpha', title=title,
                draw_averages=True, figsize=(6, 6), dpi=100)
            plt.savefig(f"plots/{dataset}/corrupted/reliability_diagrams/"
                        f"{method}_{corruption_type}.pdf")
            plt.show()

            # Draw uncertainty diagram for corruption type
            plt.figure()
            if corruption_type == 'all':
                title = f"Uncertainty Diagram on {dataset}--C\n" \
                        f"{method}"
            else:
                title = f"Uncertainty Diagram on {dataset}--C\n" \
                        f"(c = {corruption_type.title()})\n" \
                        f"{method}"
            uncertainty_diagram(
                merge_bin_data([corrupted_bin_data[(severity, corruption_type)] for severity in SEVERITY_LEVELS]),
                draw_uce=True, draw_muce=True, draw_bin_importance='alpha', title=title,
                draw_averages=True, figsize=(6, 6), dpi=100)
            plt.savefig(f"plots/{dataset}/corrupted/uncertainty_diagrams/"
                        f"{method}_{corruption_type}.pdf")
            plt.show()


def plot_shift_intensity_diagram(runs: Union[List[dict], dict],
                                 plot_values=('NLL', 'Accuracy', 'ECE', 'UCE', 'MCE', 'MUCE', 'Top-5 Accuracy',
                                              'SCE', 'ACE', 'BS', 'Model Uncertainty', 'Predictive Uncertainty')
                                 ) -> None:
    corrupted_results_df = collect_corrupted_results_df(runs)
    ngd = (corrupted_results_df.Method.str.startswith('NGD') & corrupted_results_df.Method.str.contains('k = ') &
           (corrupted_results_df.M == 1) & (corrupted_results_df.gamma == 1.0))
    gmm = (corrupted_results_df.Method.str.startswith('GMM') & corrupted_results_df.Method.str.contains('M = 1'))
    baseline = corrupted_results_df.Method.isin(['BBB', 'LL Dropout', 'Dropout', 'Deep Ensemble', 'Vanilla', 'Temp Scaling'])
    corrupted_results_df = corrupted_results_df[ngd | gmm | baseline]
    corrupted_results_df = corrupted_results_df.sort_values(['Structure', 'k', 'Method'],
                                                            ascending=[True, True, False])
    corrupted_results_df['Method'] = corrupted_results_df['Method'].apply(lambda x: x.replace(', $M = 1$', ''))

    ncol = 2
    for dataset in corrupted_results_df['Dataset'].unique():
        os.makedirs(f"plots/{dataset}/corrupted/shift_intensity", exist_ok=True)
        # plot corruption errors per dataset and optimizer grouped by model and error (ECE, accuracy, etc.)
        # 2 x len(plot_values) grid of plots
        n_methods = len(corrupted_results_df.Method.unique())
        palette = sns.color_palette("tab10", n_methods)
        for model in set(corrupted_results_df[corrupted_results_df['Dataset'] == dataset]['Model']):
            sub_df = corrupted_results_df[(corrupted_results_df['Dataset'] == dataset) &
                                          (corrupted_results_df['Model'] == model)].copy()
            for i, corruption_type in enumerate(CORRUPTION_TYPES.keys()):
                for j, value in enumerate(plot_values):
                    num_methods = len(sub_df['Method'].unique())
                    fig, ax = plt.subplots(figsize=(20, 6))
                    if corruption_type == 'all':
                        title = f"{value} on {dataset} for {model}"
                        sns.boxplot(data=sub_df, x='severity', y=value, hue='Method', ax=ax, showfliers=False, palette=palette)
                    else:
                        title = f"{value} on {dataset} for {model} on Corruption Type '{corruption_type.title()}'"
                        sns.boxplot(data=sub_df[sub_df['corruption_type'].isin(['clean', corruption_type])],
                                    x='severity', y=value, hue='Method', ax=ax, showfliers=False, palette=palette)

                    lines_per_boxplot = 5
                    num_children = 7
                    for k, box in enumerate(ax.get_children()[::num_children]):
                        if k == num_methods:
                            break
                        height = box.get_height()
                        if height == 0:
                            line = ax.lines[k * lines_per_boxplot + 4]  # the median
                            color = box.get_facecolor()
                            line.set_color(color)
                            w = line.get_linewidth()
                            line.set_linewidth(2 * w)
                    labels = [item.get_text() for item in ax.get_xticklabels()]
                    labels[0] = 'Test'
                    ax.set_xticklabels(labels)

                    ax.set_title(title, fontsize=24)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=16, ncol=ncol)
                    if value in ['ECE', 'MCE', 'UCE', 'MUCE', 'ACE', 'SCE', 'Accuracy', 'Top-5 Accuracy']:
                        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100 * x))
                        ax.yaxis.set_major_formatter(ticks)
                        plt.ylabel(rf"{value} (\%)")

                    plt.tight_layout()
                    plt.savefig(f"plots/{dataset}/corrupted/shift_intensity/{corruption_type}_{value}.pdf")
                    plt.show()
                    # for parameter in parameters:
                    #     plt.figure()
                    #     if type == 'all':
                    #         plot = sns.catplot(data=sub_df, x=parameter, y=value, hue='structure', errorbar='sd', kind='box', legend=False)
                    #     else:
                    #         plot = sns.catplot(data=sub_df[sub_df['corruption_type'].isin(['clean', type])],
                    #                     x=parameter, y=value, hue='structure', errorbar='sd', kind='box', legend=False)
                    #     title = f"Corruption Errors on {dataset} for {model} on Corruption Type '{type.title()}'"
                    #     plot.ax.set_title(title)
                    #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    #     plot.fig.tight_layout()
                    #     plt.savefig(f"plots/{dataset}/corrupted/results_parameters/{type}_{parameter}_{value}.pdf")
                    #     plt.show()


# mCE, Rel. mCE for different methods
def plot_robustness(runs: Union[List[dict], dict]) -> None:
    runs = load_all_runs(method_unique=True)
    corruption_errors = collect_corruption_errors(runs)
    datasets = list(set([run['dataset'] for run in runs]))

    # Plot corruption errors and relative corruption errors as a function of optimizer accuracy
    for dataset in datasets:
        os.makedirs(f"plots/{dataset}/corrupted/robustness", exist_ok=True)
        os.makedirs(f"plots/{dataset}/corrupted/robustness_parameters/", exist_ok=True)
        df = corruption_errors[corruption_errors.Dataset == dataset].copy().sort_values(by='Method')
        ngd = (df.Method.str.startswith('NGD') & df.Method.str.contains('k = ') & (df.M == 1) & (df.gamma == 1.0))
        gmm = (df.Method.str.startswith('GMM') & df.Method.str.contains('M = 1'))
        baseline = df.Method.isin(['BBB', 'LL Dropout', 'Dropout', 'Deep Ensemble', 'Vanilla', 'Temp Scaling'])
        df = df[ngd | gmm | baseline]
        df['Method'] = df['Method'].apply(lambda x: x.replace(', $M = 1$', ''))
        vanilla_accuracy = df[df.Method == 'Vanilla'].Accuracy.iloc[0]
        for corruption_type in CORRUPTION_TYPES.keys():
            fig, ax = plt.subplots()
            if corruption_type == 'all':
                sub_df = df.copy().melt(id_vars=['Method', 'Accuracy'], value_vars=['mCE', 'Rel. mCE'])
                title = f"Corruption Errors on {dataset}"
                sns.lineplot(sub_df, y='value', x='Accuracy', style='variable', hue='variable')
                sns.scatterplot(sub_df.groupby(['Method', 'variable']).mean().reset_index().sort_values(by='Accuracy'),
                                hue='Method', y='value', x='Accuracy')
                plt.axvline(x=vanilla_accuracy, color='black', linestyle=':')
                plt.axhline(y=1.0, color='black', linestyle=':')
            else:
                sub_df = df[df.Type == corruption_type].copy().\
                    melt(id_vars=['Method', 'Accuracy'], value_vars=['mCE', 'Rel. mCE'])
                title = f"Corruption Errors on {dataset} for Corruption Type '{corruption_type.title()}'"
                sns.lineplot(sub_df, y='value', x='Accuracy', style='variable', hue='variable')
                sns.scatterplot(sub_df.groupby(['Method', 'variable']).mean().reset_index().sort_values(by='Accuracy'),
                                hue='Method', y='value', x='Accuracy')
                plt.axvline(x=vanilla_accuracy, color='black', linestyle=':')
                plt.axhline(y=1.0, color='black', linestyle=':')
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100 * x))
            ax.xaxis.set_major_formatter(ticks)
            ax.yaxis.set_major_formatter(ticks)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
            plt.xlabel(r'Accuracy (\%)')
            plt.ylabel(r'\%')
            plt.title(title, fontsize=20)
            plt.tight_layout()
            plt.savefig(f"plots/{dataset}/corrupted/robustness/mCE_{corruption_type}.pdf")
            plt.show()

            # plt.figure()
            # if corruption_type == 'all':
            #     sns.scatterplot(sub_df, hue='Method', y='Rel. mCE', x='Accuracy')
            # else:
            #     sns.scatterplot(sub_df[sub_df.Type == corruption_type], hue='Method', y='Rel. mCE', x='Accuracy')
            # plt.title(f"mCE on {dataset} for Corruption Type '{corruption_type.title()}'")
            # plt.savefig(f"plots/{dataset}/corrupted/robustness/rmCE_{corruption_type}.pdf")
            # plt.show()


def plot_uncertainty(runs: Union[List[dict], dict]) -> None:
    runs = load_all_runs(method_unique=True)
    datasets = list(set([run['dataset'] for run in runs]))
    uncertainty = collect_uncertainty(runs)
    for dataset in datasets:
        os.makedirs(f"plots/{dataset}", exist_ok=True)

        g = sns.FacetGrid(uncertainty[(uncertainty.Dataset == dataset)],
                          col='Method', row='severity', margin_titles=True, hue='severity')
        g.map(sns.distplot, 'Predictive Uncertainty')
        g.add_legend()
        plt.savefig(f"plots/{dataset}/predictive_uncertainty.pdf")

        # g = sns.FacetGrid(uncertainty[
        #                       uncertainty.Dataset == dataset &
        #                       (~uncertainty.Method.isin(['Vanilla', 'Temp Scaling']))],
        #                   col='Method', hue='severity', margin_titles=True)
        # g.map(sns.kdeplot, 'Model Uncertainty', bw_adjust=0.2, clip=(0, np.log(num_classes)))
        # plt.savefig(f"plots/{dataset}/model_uncertainty.pdf")


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

    runs = load_all_runs(method_unique=True)
    plot_runs(runs)
