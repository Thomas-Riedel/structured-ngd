from typing import Union, List
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import numpy as np
import pandas as pd
from optimizers.noisy_optimizer import NoisyOptimizer
import pickle


SEVERITY_LEVELS = [1, 2, 3, 4, 5]

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


def load_run(dataset, model, optimizer, directory: str = 'runs') -> dict:
    if type(model) != str:
        model = model.__name__
    if type(optimizer) != str:
        optimizer = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                run = pickle.load(f)
            if ((run['optimizer_name'].lower() == optimizer.lower()) and
                    (run['model_name'].lower() == model.lower()) and
                    (run['dataset'].lower() == dataset.lower())):
                return run
    return None


def load_all_runs(directory: str = 'runs') -> List[dict]:
    runs = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                runs.append(pickle.load(f))
    return runs


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


def get_corrupted_results(dataset, model, optimizer, baseline, metrics, clean_results, mc_samples, n_bins):
    if not dataset.lower() in ['cifar10', 'cifar100']:
        return None
    model_name = model.__name__
    optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__

    severity = 0
    corruption = 'clean'
    corruption_type = 'clean'

    df = pd.DataFrame([dict(
        corruption_type=corruption_type,
        corruption=corruption,
        severity=severity,
        loss=clean_results['test_loss'],
        **clean_results['test_metrics']
    )])
    bin_data = {(severity, corruption_type): clean_results['bin_data']}
    i = 0
    for severity in SEVERITY_LEVELS:
        for corruption_type in CORRUPTION_TYPES.keys():
            if corruption_type == 'all':
                continue
            bin_data_list = []
            for corruption in CORRUPTION_TYPES[corruption_type]:
                data_loader = load_corrupted_data(dataset, corruption, severity)
                loss, metric, corrupted_bin_data = model.evaluate(data_loader, metrics=metrics, optimizer=optimizer,
                                                                  mc_samples=mc_samples, n_bins=n_bins)
                bin_data_list.append(corrupted_bin_data)
                corrupted_result = pd.DataFrame([dict(
                        corruption_type=corruption_type,
                        corruption=corruption,
                        severity=severity,
                        loss=loss,
                        **metric
                    )])
                df = pd.concat([df, corrupted_result], ignore_index=True)
                i += 1
                print(f"[{i} / {len(SEVERITY_LEVELS) * len(CORRUPTIONS)}]; "
                      f"severity = {severity}, corruption = {corruption}")

            # group bin_data results by types
            corrupted_bin_data = merge_bin_data(bin_data_list)
            bin_data[(severity, corruption_type)] = corrupted_bin_data
            bin_data[(severity, 'all')] = corrupted_bin_data

    sub_df = df[
        (df['severity'] > 0) & (df['corruption_type'] != 'all')
    ].drop(['corruption_type', 'severity'], axis=1).copy()
    clean_accuracy = clean_results['test_metrics']['accuracy']
    if isinstance(optimizer, NoisyOptimizer):
        baseline_results = load_run(dataset, model, baseline)
        baseline_clean_accuracy = baseline_results['test_metrics']['accuracy']
        baseline_df = baseline_results['corrupted_results']['df']
        baseline_df = baseline_df[
            (baseline_df['severity'] > 0) & (df['corruption_type'] != 'all')
        ].drop(['corruption_type', 'severity'], axis=1).copy()
    else:
        baseline_clean_accuracy = clean_accuracy
        baseline_df = sub_df.copy()
    corruption_error = ce(sub_df, baseline_df)
    rel_corruption_error = rel_ce(sub_df, baseline_df, clean_accuracy, baseline_clean_accuracy)

    for corruption_type in CORRUPTION_TYPES.keys():
        corruption_error[corruption_type] = corruption_error[CORRUPTION_TYPES[corruption_type]].mean(1)
        corruption_error[f"{corruption_type}_std"] = corruption_error[CORRUPTION_TYPES[corruption_type]].std(1)

        rel_corruption_error[corruption_type] = rel_corruption_error[CORRUPTION_TYPES[corruption_type]].mean(1)
        rel_corruption_error[f"{corruption_type}_std"] = rel_corruption_error[CORRUPTION_TYPES[corruption_type]].std(1)

    df['dataset'] = dataset
    df['model_name'] = model_name
    df['optimizer_name'] = optimizer_name

    corruption_error['dataset'] = dataset
    corruption_error['model_name'] = model_name
    corruption_error['optimizer_name'] = optimizer_name
    corruption_error['accuracy'] = clean_results['test_metrics']['accuracy']

    rel_corruption_error['dataset'] = dataset
    rel_corruption_error['model_name'] = model_name
    rel_corruption_error['optimizer_name'] = optimizer_name
    rel_corruption_error['accuracy'] = clean_results['test_metrics']['accuracy']

    corrupted_results = dict(
        model_name=model_name,
        optimizer_name=optimizer_name,
        dataset=dataset,
        corruption_error=corruption_error,
        rel_corruption_error=rel_corruption_error,
        df=df,
        bin_data=bin_data
    )
    return corrupted_results


def ce(df, baseline_corrupted_df):
    result = pd.DataFrame()
    for corruption in CORRUPTIONS:
        sub_df = df[df.corruption == corruption]
        sub_baseline_df = baseline_corrupted_df[baseline_corrupted_df.corruption == corruption]
        result[corruption] = [np.sum(1 - sub_df.accuracy) / np.sum(1 - sub_baseline_df.accuracy)]
    return result


def rel_ce(df, baseline_corrupted_df, clean_accuracy, baseline_clean_accuracy):
    result = pd.DataFrame()
    for corruption in CORRUPTIONS:
        sub_df = df[df.corruption == corruption]
        sub_baseline_df = baseline_corrupted_df[baseline_corrupted_df.corruption == corruption]
        result[corruption] = [np.sum(clean_accuracy - sub_df.accuracy) / np.sum(baseline_clean_accuracy - sub_baseline_df.accuracy)]
    return result


def collect_corrupted_results_df(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corrupted_results_df = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        dataset = corrupted_results['dataset']
        params = run['params']
        if not run['optimizer_name'].startswith('StructuredNGD'):
            optimizer_name = run['optimizer_name']
        else:
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer_name = rf"NGD (structure = {structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']})$"
        corrupted_results['df']['optimizer_name'] = optimizer_name
        clean_df = pd.DataFrame([dict(
                dataset=dataset,
                model_name=run['model_name'],
                optimizer_name=optimizer_name,
                corruption_type='clean',
                corruption='clean',
                severity=0,
                loss=run['test_loss'],
                **run['test_metrics']
            )])
        corrupted_results_df = pd.concat([corrupted_results_df, clean_df, corrupted_results['df']], ignore_index=True)
    corrupted_results_df.rename(columns={'optimizer_name': 'optimizer', 'model_name': 'model', 'accuracy': 'Accuracy',
                                         'top_5_accuracy': 'Top-5 Accuracy', 'ece': 'ECE', 'mce': 'MCE',
                                         'loss': 'Loss'}, inplace=True)
    return corrupted_results_df


def collect_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corruption_errors = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corruption_error = run['corrupted_results']['corruption_error']
        corruption_error['dataset'] = run['dataset']
        corruption_error['optimizer_name'] = run['optimizer_name'] \
            if not run['optimizer_name'].startswith('StructuredNGD') else \
            rf"NGD $(k = {run['params']['k']}, M = {run['params']['mc_samples']}, \gamma = {run['params']['gamma']})$"
        corruption_errors = pd.concat([corruption_errors, corruption_error], ignore_index=True)
    return corruption_errors


def collect_rel_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    rel_corruption_errors = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        corrupted_results['rel_corruption_error']['dataset'] = run['dataset']
        corrupted_results['rel_corruption_error']['optimizer_name'] = run['optimizer_name'] \
            if not run['optimizer_name'].startswith('StructuredNGD') else \
            f"k = {run['params']['k']}, M = {run['params']['mc_samples']}"
        rel_corruption_errors = pd.concat([rel_corruption_errors, corrupted_results['rel_corruption_error']],
                                          ignore_index=True)
    return rel_corruption_errors


def merge_bin_data(data: List[dict]):
    if len(data) == 0:
        return {}
    result = data[0]
    bins = result['bins']
    bin_accuracies = result['counts'] * result['accuracies']
    bin_confidences = result['counts'] * result['confidences']
    bin_counts = result['counts']

    for bin_data in data[1:]:
        assert(len(bins) == len(bin_data['bins']))
        bin_accuracies += bin_data['counts'] * bin_data['accuracies']
        bin_confidences += bin_data['counts'] * bin_data['confidences']
        bin_counts += bin_data['counts']

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
