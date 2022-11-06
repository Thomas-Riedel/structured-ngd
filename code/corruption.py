from typing import Union, List
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import numpy as np
import pandas as pd
from optimizers.noisy_optimizer import NoisyOptimizer
from network import TempScaling, DeepEnsemble, HyperDeepEnsemble
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


def load_run(dataset, model, optimizer, method='Vanilla', directory: str = 'runs') -> dict:
    if type(model) != str:
        model = model.__name__
    if type(optimizer) != str:
        optimizer = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
    for file in sorted(os.listdir(directory)):
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                run = pickle.load(f)
            if ((run['optimizer_name'].lower() == optimizer.lower()) and
                    (run['model_name'].lower() == model.lower()) and
                    (run['dataset'].lower() == dataset.lower()) and
                    (run['method'] == method)
            ):
                return run
    return None


def load_all_runs(directory: str = 'runs') -> List[dict]:
    runs = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                run = pickle.load(f)
            run['total_time'] = run['epoch_times'][-1]
            if not 'num_epochs' in run:
                run['num_epochs'] = len(run['epoch_times']) - 1
                run['avg_time_per_epoch'] = run['total_time'] / run['num_epochs']
            if run['params'].get('k') == 0:
                run['params']['structure'] = 'diagonal'
            runs.append(run)
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
        corrupted_input = corrupted_input[(severity - 1) * data_size:severity * data_size] / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1))
        corrupted_input = (corrupted_input - mean) / std

        data.append(TensorDataset(corrupted_input, labels))
    data = ConcatDataset(data)
    # define dataloader
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=2)
    return data_loader


def get_corrupted_results(dataset, model, optimizer, method, baseline, metrics, clean_results, mc_samples, n_bins):
    if not dataset.lower() in ['cifar10', 'cifar100']:
        return None
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

    sub_df = df[
        (df['severity'] > 0) & (df['corruption_type'] != 'all')
    ].drop(['corruption_type', 'severity'], axis=1).copy()
    clean_accuracy = clean_results['test_metrics']['accuracy']
    if isinstance(optimizer, NoisyOptimizer) or method in ['BBB', 'Dropout', 'LL Dropout']:
        baseline_results = load_run(dataset, model, baseline)
        baseline_clean_accuracy = baseline_results['test_metrics']['accuracy']
        baseline_df = baseline_results['corrupted_results']['df']
        baseline_df = baseline_df[
            (baseline_df['severity'] > 0) & (df['corruption_type'] != 'all')
        ].drop(['corruption_type', 'severity'], axis=1).copy()
    else:
        if isinstance(model, (TempScaling, DeepEnsemble, HyperDeepEnsemble)):
            baseline_results = load_run(dataset, model.model, baseline)
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
        method=method,
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


def collect_results(runs, directory='runs', baseline='SGD'):
    results = pd.DataFrame(columns=['Method', 'Dataset', 'Model', 'Optimizer', 'Structure', 'k', 'M', 'gamma',
                                    'Training Loss', 'Test Loss', 'Test Accuracy', 'Top-k Accuracy',
                                    'ECE', 'MCE',
                                    # 'Total Time (h)', 'Avg. Time per Epoch (s)',
                                    'Num Epochs'])
    for run in runs:
        method = run['method']
        dataset = run['dataset']
        model = run['model_name']
        if run['optimizer_name'].startswith('StructuredNGD'):
            params = run['params']
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer = f"NGD (structure = {structure}, $k = {params['k']}, " \
                        f"M = {params['mc_samples']}, \gamma = {params['gamma']}$)"
        else:
            optimizer = run['optimizer_name']
        baseline_run = load_run(dataset, model, baseline, directory=directory)
        train_loss = run['train_loss'][-1]
        val_loss = run['val_loss'][-1]
        test_loss = run['test_loss']
        test_accuracy = run['test_metrics']['accuracy']
        top_k_accuracy = run['test_metrics']['top_5_accuracy']
        ece = run['bin_data']['expected_calibration_error']
        mce = run['bin_data']['max_calibration_error']
        # total_time = run['total_time']
        # avg_time_per_epoch = run['avg_time_per_epoch']
        num_epochs = run['num_epochs']
        if run['optimizer_name'] == baseline and method == 'Vanilla':
            train_loss = compare(train_loss)
            val_loss = compare(val_loss)
            test_loss = compare(test_loss)
            test_accuracy = compare(100 * test_accuracy)
            top_k_accuracy = compare(100 * top_k_accuracy)
            ece = compare(100 * ece)
            mce = compare(100 * mce)
            # total_time = compare(total_time / 3600)
            # avg_time_per_epoch = compare(avg_time_per_epoch)
        else:
            train_loss = compare(train_loss, baseline_run['train_loss'][-1])
            val_loss = compare(val_loss, baseline_run['val_loss'][-1])
            test_loss = compare(test_loss, baseline_run['test_loss'])
            test_accuracy = compare(100 * test_accuracy, 100 * baseline_run['test_metrics']['accuracy'])
            top_k_accuracy = compare(100 * top_k_accuracy, 100 * baseline_run['test_metrics']['top_5_accuracy'])
            ece = compare(100 * ece, 100 * baseline_run['bin_data']['expected_calibration_error'])
            mce = compare(100 * mce, 100 * baseline_run['bin_data']['max_calibration_error'])
            # total_time = compare(total_time / 3600, baseline_run['total_time'] / 3600)
            # avg_time_per_epoch = compare(avg_time_per_epoch, baseline_run['avg_time_per_epoch'])

        if run['optimizer_name'].startswith('StructuredNGD'):
            structure = run['params']['structure'].replace('_', ' ').title().replace(' ', '')
            k = run['params']['k']
            M = run['params']['mc_samples']
            gamma = run['params']['gamma']
        else:
            structure = '--'
            k = '--'
            M = '--'
            gamma = '--'

        result = pd.DataFrame([{
            'Method': method,
            'Dataset': dataset.upper(),
            'Model': model,
            'Optimizer': optimizer,
            'Structure': structure,
            'k': k,
            'M': M,
            'gamma': gamma,
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy,
            'Top-k Accuracy': top_k_accuracy,
            'ECE': ece, 'MCE': mce,
            # 'Total Time (h)': total_time,
            # 'Avg. Time per Epoch (s)': avg_time_per_epoch,
            'Num Epochs': num_epochs
        }])
        results = pd.concat([results, result], ignore_index=True)
    results.sort_values(by=['Dataset', 'Model', 'Method', 'Optimizer'], inplace=True)
    return results


def collect_corrupted_results_df(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corrupted_results_df = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        corrupted_results = run['corrupted_results']
        method = run['method']
        dataset = corrupted_results['dataset']
        params = run['params']
        if not run['optimizer_name'].startswith('StructuredNGD'):
            optimizer_name = run['optimizer_name']
            structure = '--'
            k = '--'
            M = '--'
            gamma = '--'
        else:
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            optimizer_name = rf"NGD (structure = {structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']})$"
            structure = run['params']['structure'].replace('_', ' ').title().replace(' ', '')
            k = run['params']['k']
            M = run['params']['mc_samples']
            gamma = run['params']['gamma']
        corrupted_results['df']['model_name'] = run['model_name']
        corrupted_results['df']['method'] = method
        corrupted_results['df']['optimizer_name'] = optimizer_name
        corrupted_results['df']['structure'] = structure
        corrupted_results['df']['k'] = k
        corrupted_results['df']['M'] = M
        corrupted_results['df']['gamma'] = gamma

        clean_df = pd.DataFrame([dict(
                method=method,
                dataset=dataset,
                model_name=run['model_name'],
                optimizer_name=optimizer_name,
                structure=structure,
                k=k,
                M=M,
                gamma=gamma,
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
        corruption_error['method'] = run['method']
        corruption_error['dataset'] = run['dataset']
        params = run['params']
        if not run['optimizer_name'].startswith('StructuredNGD'):
            corruption_error['optimizer_name'] = run['optimizer_name']
            corruption_error['Structure'] = '--'
            corruption_error['k'] = '--'
            corruption_error['M'] = '--'
            corruption_error['gamma'] = '--'
        else:
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            corruption_error['optimizer_name'] = rf"NGD (structure = {structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']})$"
            corruption_error['Structure'] = structure
            corruption_error['k'] = params['k']
            corruption_error['M'] = params['mc_samples']
            corruption_error['gamma'] = params['gamma']
        for key, value in run['test_metrics'].items():
            corruption_error[key] = value
        corruption_errors = pd.concat([corruption_errors, corruption_error], ignore_index=True)
    return corruption_errors


def collect_rel_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    rel_corruption_errors = pd.DataFrame()
    for run in runs:
        if not run['dataset'].lower() in ['cifar10', 'cifar100']:
            continue
        rel_corruption_error = run['corrupted_results']['rel_corruption_error']
        rel_corruption_error['method'] = run['method']
        rel_corruption_error['dataset'] = run['dataset']
        params = run['params']
        if not run['optimizer_name'].startswith('StructuredNGD'):
            rel_corruption_error['optimizer_name'] = run['optimizer_name']
            rel_corruption_error['Structure'] = '--'
            rel_corruption_error['k'] = '--'
            rel_corruption_error['M'] = '--'
            rel_corruption_error['gamma'] = '--'
        else:
            structure = params['structure'].replace('_', ' ').title().replace(' ', '')
            rel_corruption_error['Structure'] = structure
            rel_corruption_error['k'] = params['k']
            rel_corruption_error['M'] = params['mc_samples']
            rel_corruption_error['gamma'] = params['gamma']
            rel_corruption_error['optimizer_name'] = rf"NGD (structure = {structure}, $k = {params['k']}, M = {params['mc_samples']}, \gamma = {params['gamma']})$"
        for key, value in run['test_metrics'].items():
            rel_corruption_error[key] = value
        rel_corruption_errors = pd.concat([rel_corruption_errors, rel_corruption_error], ignore_index=True)
    return rel_corruption_errors


def compare(x, y=None, f='{:.1f}'):
    if y is None:
        return f.format(x)
    sign = ['+', '±', ''][1 - np.sign(x - y).astype(int)]
    relative_diff = 100 * (x / y - 1)
    return f"{f.format(x)} ({sign}{f.format(relative_diff)}%)"


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
