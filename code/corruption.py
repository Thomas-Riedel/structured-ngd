from typing import Union, List
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import numpy as np
import pandas as pd
from optimizers import NoisyOptimizer
from plot_runs import load_run


CORRUPTIONS = [
    'brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate',
    'snow', 'speckle_noise', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'pixelate',
    'shot_noise', 'spatter', 'zoom_blur'
]

CORRUPTION_TYPES = dict(
    all=CORRUPTIONS,
    noise=['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
    blur=['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur'],
    weather=['snow', 'frost', 'fog', 'brightness', 'spatter'],
    digital=['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate']
)


def load_corrupted_data(dataset: str, corruption: Union[str, List[str]], severity: int, batch_size: int = 128):
    path = '/storage/group/dataset_mirrors/01_incoming'
    if dataset.lower() == 'cifar10':
        data_name = 'CIFAR-10-C'
    elif dataset.lower() == 'cifar100':
        data_name = 'CIFAR-100-C'
    else:
        raise ValueError()
    data_size = 10000

    # load labels
    directory = os.path.join(path, data_name, 'data')
    path_to_file = os.path.join(directory, 'labels.npy')
    labels = torch.from_numpy(np.load(path_to_file))[:data_size]

    if type(corruption) == str:
        corruption = [corruption]

    data = []
    for c in corruption:
        path_to_file = os.path.join(directory, f"{c}.npy")

        # load corrupted inputs and preprocess
        corrupted_input = torch.from_numpy(np.load(path_to_file)).permute(0, 3, 1, 2).float()
        corrupted_input = corrupted_input[(severity - 1) * data_size:severity * data_size]
        max = corrupted_input.max()
        min = corrupted_input.min()
        corrupted_input /= (max - min)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1))
        corrupted_input = (corrupted_input - mean) / std

        data.append(TensorDataset(corrupted_input, labels))
    data = ConcatDataset(data)
    # define dataloader
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=4)
    return data_loader


def get_corrupted_results(dataset, model, optimizer, metrics, clean_results, mc_samples, n_bins):
    if not dataset.lower() in ['cifar10', 'cifar100']:
        return None
    severity = 0
    corruption = 'clean'
    corruption_type = 'clean'

    df = pd.DataFrame(dict(
        corruption_type=corruption_type,
        corruption=corruption,
        severity=severity,
        loss=clean_results['test_loss'],
        **clean_results['test_metrics']
    ))
    bin_data = {(severity, corruption_type): clean_results['bin_data']}
    for severity in [1, 2, 3, 4, 5]:
        for corruption_type in CORRUPTION_TYPES.keys():
            bin_data_list = []
            for corruption in CORRUPTION_TYPES[corruption_type]:
                data_loader = load_corrupted_data(dataset, corruption, severity)
                loss, metrics = model.evaluate(data_loader, metrics=metrics,
                                               optimizer=optimizer, mc_samples=mc_samples)
                corrupted_bin_data = model.compute_calibration(data_loader, n_bins=n_bins,
                                                               optimizer=optimizer, mc_samples=mc_samples)
                bin_data_list.append(corrupted_bin_data)
                corrupted_result = pd.DataFrame(
                    dict(
                        corruption_type=corruption_type,
                        corruption=corruption,
                        severity=severity,
                        loss=loss,
                        **metrics
                    )
                )
                df = df.append(corrupted_result)

            # group bin_data results by types
            corrupted_bin_data = merge_bin_data(bin_data_list)
            bin_data[(severity, corruption_type)] = corrupted_bin_data

    clean_accuracy = clean_results['test_accuracy']
    if not isinstance(optimizer, NoisyOptimizer):
        adam_clean_results = load_run(dataset, model, optimizer)
        adam_accuracy = adam_clean_results['test_metrics']['accuracy']
    else:
        adam_accuracy = clean_accuracy

    sub_df = df[df['severity'] > 0].drop('severity', axis=1).copy()
    corr_error = sub_df.groupby(['corruption_type', 'corruption']).agg(
        corruption_error(adam_accuracy)
    )
    rel_corr_error = sub_df.groupby(['corruption_type', 'corruption']).agg(
        relative_corruption_error(adam_accuracy, clean_accuracy)
    )

    mce = corr_error.copy().groupby('corruption_type').agg(['mean', 'std'])
    rmce = rel_corr_error.copy().groupby('corruption_type').agg(['mean', 'std'])

    corr_error['dataset'] = dataset
    corr_error['model_name'] = model.__name__
    corr_error['optimizer_name'] = optimizer.__name__
    corr_error['accuracy'] = clean_results['test_accuracy']

    rel_corr_error['dataset'] = dataset
    rel_corr_error['model_name'] = model.__name__
    rel_corr_error['optimizer_name'] = optimizer.__name__
    rel_corr_error['accuracy'] = clean_results['test_accuracy']

    mce['dataset'] = dataset
    mce['model_name'] = model.__name__
    mce['optimizer_name'] = optimizer.__name__
    mce['accuracy'] = clean_results['test_accuracy']

    rmce['dataset'] = dataset
    rmce['model_name'] = model.__name__
    rmce['optimizer_name'] = optimizer.__name__
    rmce['accuracy'] = clean_results['test_accuracy']

    corrupted_results = dict(
        model_name=model.__name__,
        optimizer_name=optimizer.__name__,
        dataset=dataset,
        corruption_error=corr_error,
        rel_corruption_error=rel_corr_error,
        mce=mce,
        rmce=rmce,
        df=df,
        bin_data=bin_data
    )
    return corrupted_results


def corruption_error(adam_accuracy):
    def f(df):
        if df.corruption == 'clean':
            return 0
        else:
            np.sum(1 - df.accuracy) / np.sum(1 - adam_accuracy)
    return f


def relative_corruption_error(adam_accuracy, clean_accuracy):
    def f(df):
        if df.corruption == 'clean':
            return 0
        else:
            (np.sum(1 - df.accuracy) - (1 - clean_accuracy)) / (np.sum(1 - adam_accuracy) - (1 - clean_accuracy))
        return f


def collect_corrupted_results_df(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corrupted_results_df = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue

        corrupted_results = run['corrupted_results']
        dataset = corrupted_results['dataset']

        clean_df = pd.DataFrame(
            dict(
                dataset=dataset,
                model=run['model_name'],
                optimizer=run['optimizer_name'],
                corruption_type='clean',
                corruption='clean',
                severity=0,
                loss=run['test_loss'],
                **run['test_metrics']
            )
        )
        corrupted_results_df = corrupted_results_df.append(clean_df)
        corrupted_results_df = corrupted_results_df.append(corrupted_results['df'])
    corrupted_results_df.rename(columns={'optimizer_name': 'optimizer', 'model_name': 'model'}, inplace=True)
    return corrupted_results_df


def collect_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corruption_errors = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        corruption_errors = corruption_errors.append(corrupted_results['corruption_error'])
    return corruption_errors


def collect_rel_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    rel_corruption_errors = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        rel_corruption_errors = rel_corruption_errors.append(corrupted_results['rel_corruption_error'])
    return rel_corruption_errors


def collect_mce(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    mce = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        mce = mce.append(corrupted_results['mce'])
    return mce


def collect_rmce(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    rmce = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        rmce = rmce.append(corrupted_results['rmce'])
    return rmce


def merge_bin_data(data: List[dict]):
    if len(data) == 0:
        return {}
    result = data[0]
    bins = result['bins']
    bin_accuracies = result['bin_counts'] * result['accuracies']
    bin_confidences = result['bin_counts'] * result['confidences']
    bin_counts = result['bin_counts']

    for bin_data in data[1:]:
        assert(len(bins) == len(bin_data['bins']))
        bin_accuracies += bin_data['bin_counts'] * bin_data['accuracies']
        bin_confidences += bin_data['bin_counts'] * bin_data['confidences']
        bin_counts += bin_data['bin_counts']

    bin_accuracies /= bin_counts
    bin_confidences /= bin_counts

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
