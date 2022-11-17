import torch.utils.data
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CIFAR100, STL10, SVHN, ImageNet, ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.optim import *
import numpy as np

import pandas as pd
import os, pickle, argparse, sys
from typing import Union, List, Tuple, Any
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import re
import itertools

from models import *
from metrics import *
from optimizers.noisy_optimizer import *
from optimizers.rank_k_cov import *
from network import Model


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
                        help='Optimizer, one of Adam, SGD, StructuredNGD '
                             '(capitalization matters!, default: StructuredNGD)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train models on data (default: 1)')
    parser.add_argument('-d', '--dataset', type=str, default="CIFAR10",
                        help='Dataset for training, one of CIFAR10, CIFAR100, MNIST, FashionMNIST (default: CIFAR10)')
    parser.add_argument('-m', '--model', type=str, default="ResNet32",
                        help='ResNet model (default: ResNet32)')
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
    parser.add_argument('--mc_samples_val', type=int, default=8,
                        help='Number of MC samples during evaluation (default: 8)')
    parser.add_argument('--mc_samples_test', type=int, default=64,
                        help='Number of MC samples during testing (default: 64)')
    parser.add_argument('--baseline', type=str, default='SGD',
                        help='Baseline optimizer for comparison of corrupted data')
    parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(),
                        help='Whether to use CUDA (default: True if available, False if not)')

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
        mc_samples_val=args.mc_samples_val,
        mc_samples_test=args.mc_samples_test,
        structure=args.structure,
        eval_every=args.eval_every,
        momentum_grad=args.momentum_grad,
        momentum_prec=args.momentum_prec,
        prior_precision=args.prior_precision,
        damping=args.damping,
        gamma=args.gamma,
        data_split=args.data_split,
        n_bins=args.n_bins,
        baseline=args.baseline,
        use_cuda=args.use_cuda
    )
    return args_dict


def parse_vals(args: List[str], types: List[type]) -> List[Union[int, float]]:
    """Parse string values from command line, separate and form into list.

    :param args: List[str], list of supplied command line arguments as strings
    :param types: List[type], list of specified types to parse supplied arguments to
    :return: result, List[Union[int, float]], list of parsed and separated arguments
    """

    def make_sequence(t):
        def f(val):
            if t == int:
                # argument can be given as "0to11step3" meaning it should be parsed into a list ranging from 0 to 11
                # including with a step size of 3, i.e. [0, 3, 6, 9]
                val_list = [t(x) for x in re.split(r'to|step', val)]
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
                return [t(val)]
        return f

    result = []
    for arg, t in zip(args, types):
        result.append(list(sorted(set(itertools.chain.from_iterable(map(make_sequence(t), arg.split(',')))))))
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

    img_size = (32, 32)
    if dataset.lower() == 'imagenet':
        img_size = (64, 64)
    elif dataset.lower() == 'stl10':
        img_size = (96, 96)
    if dataset.lower().startswith('cifar'):
        transform_augmented = CIFAR_TRANSFORM_AUGMENTED
        transform = CIFAR_TRANSFORM
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))]
        )
        transform_augmented = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_size, padding=pad_size),
            transforms.RandomHorizontalFlip(p=0.5),
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
        training_data = TinyImageNet(split='train', transform=transform_augmented)
        test_data = TinyImageNet(split='val', transform=transform)
    elif dataset.lower() == "svhn":
        training_data = SVHN('data/SVHN/train', download=True, split='train', transform=transform_augmented)
        test_data = SVHN('data/SVHN/test', download=True, split='test', transform=transform)
    else:
        raise ValueError(f"Dataset {dataset} not recognized! Choose one of "
                         f"[mnist, fmnist, cifar10, cifar100, stl10, imagenet, svhn]")

    seed = 42
    split = int(split * len(training_data))
    training_data, validation_data = torch.utils.data.random_split(training_data, [split, len(training_data) - split],
                                                                   generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(
        training_data, batch_size=batch_size, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        validation_data, batch_size=batch_size, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_params(args: dict, optimizer=SGD, add_weight_decay=True, n=1) -> dict:
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
    if optimizer.__name__ == 'Adam':
        momentum = dict(betas=(0.9, 0.999))
    elif optimizer.__name__ == 'SGD':
        momentum = dict(momentum=0.9, nesterov=True)
    else:
        momentum = dict()
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


def load_run(dataset, model, optimizer, directory: str = 'runs', method='Vanilla') -> dict:
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


def load_all_runs(directory: str = 'runs', method_unique: bool = False) -> List[dict]:
    runs = []
    methods = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".pkl"):
            with open(os.path.join(directory, file), 'rb') as f:
                run = pickle.load(f)
            if method_unique:
                if run['method'].startswith('NGD') and (run['params'].get('gamma') != 1.0 or run['params'].get('mc_samples') != 1):
                    continue
                if (run['method'], run['dataset']) in methods:
                    continue
            run['total_time'] = run['epoch_times'][-1]
            if not 'num_epochs' in run:
                run['num_epochs'] = len(run['epoch_times']) - 1
                run['avg_time_per_epoch'] = run['total_time'] / run['num_epochs']
            if run['params'].get('k') == 0:
                run['params']['structure'] = 'diagonal'
            runs.append(run)
            methods.append((run['method'], run['dataset']))
    return runs


def collect_results(runs, directory='runs', baseline='SGD'):
    results = pd.DataFrame(columns=['Method', 'Dataset', 'Model', 'Optimizer', 'Structure', 'k', 'M', 'gamma',
                                    'Training Loss', 'Validation Loss', 'Test Loss', 'Test Accuracy', 'Test Error',
                                    'BS', 'ECE', 'MCE', 'UCE', 'MUCE', 'Top-k Accuracy', 'Top-k Error',
                                    'ACE', 'SCE', 'MI', 'PU',
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
        baseline_run = load_run(dataset, model, baseline, method='Vanilla', directory=directory)
        train_loss = run['train_loss'][-1]
        val_loss = run['val_loss'][-1]
        test_loss = run['test_loss']
        test_accuracy = run['test_metrics']['accuracy']
        top_k_accuracy = run['test_metrics']['top_5_accuracy']
        test_error = 1 - run['test_metrics']['accuracy']
        top_k_error = 1 - run['test_metrics']['top_5_accuracy']
        brier_score = run['test_metrics']['brier']
        ece = run['test_metrics']['ece']
        mce = run['test_metrics']['mce']
        uce = run['test_metrics']['uce']
        muce = run['test_metrics']['muce']
        ace = run['test_metrics']['ace']
        sce = run['test_metrics']['sce']
        mi = run['test_metrics']['model_uncertainty']
        pu = run['test_metrics']['predictive_uncertainty']
        # total_time = run['total_time']
        # avg_time_per_epoch = run['avg_time_per_epoch']
        num_epochs = run['num_epochs']
        if method == 'Vanilla':
            train_loss = compare(train_loss)
            val_loss = compare(val_loss)
            test_loss = compare(test_loss)
            test_accuracy = compare(100 * test_accuracy)
            top_k_accuracy = compare(100 * top_k_accuracy)
            test_error = compare(100 * test_error)
            top_k_error = compare(100 * top_k_error)
            brier_score = compare(brier_score)
            ece = compare(100 * ece)
            mce = compare(100 * mce)
            uce = compare(100 * uce)
            muce = compare(100 * muce)
            ace = compare(100 * ace)
            sce = compare(100 * sce)
            mi = compare(mi)
            pu = compare(pu)
            # total_time = compare(total_time / 3600)
            # avg_time_per_epoch = compare(avg_time_per_epoch)
        else:
            train_loss = compare(train_loss, baseline_run['train_loss'][-1])
            val_loss = compare(val_loss, baseline_run['val_loss'][-1])
            test_loss = compare(test_loss, baseline_run['test_loss'])
            test_accuracy = compare(100 * test_accuracy, 100 * baseline_run['test_metrics']['accuracy'])
            top_k_accuracy = compare(100 * top_k_accuracy, 100 * baseline_run['test_metrics']['top_5_accuracy'])
            test_error = compare(100 * test_error, 100 * (1 - baseline_run['test_metrics']['accuracy']))
            top_k_error = compare(100 * top_k_error, 100 * (1 - baseline_run['test_metrics']['top_5_accuracy']))
            brier_score = compare(brier_score, baseline_run['test_metrics']['brier'])
            ece = compare(100 * ece, 100 * baseline_run['test_metrics']['ece'])
            mce = compare(100 * mce, 100 * baseline_run['test_metrics']['mce'])
            uce = compare(100 * uce, 100 * baseline_run['test_metrics']['uce'])
            muce = compare(100 * muce, 100 * baseline_run['test_metrics']['muce'])
            ace = compare(100 * ace, 100 * baseline_run['test_metrics']['ace'])
            sce = compare(100 * sce, 100 * baseline_run['test_metrics']['sce'])
            mi = compare(mi, baseline_run['test_metrics']['predictive_uncertainty'])
            pu = compare(pu, baseline_run['test_metrics']['predictive_uncertainty'])
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
            'Dataset': dataset,
            'Model': model,
            'Optimizer': optimizer,
            'Structure': structure,
            'k': k,
            'M': M,
            'gamma': gamma,
            'Training Loss': train_loss,
            'Validation Loss': val_loss,
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy, 'Test Error': test_error, 'BS': brier_score,
            'ECE': ece, 'MCE': mce, 'UCE': uce, 'MUCE': muce,
            'Top-k Accuracy': top_k_accuracy, 'Top-k Error': top_k_error,
            'ACE': ace, 'SCE': sce, 'MI': mi, 'PU': pu,
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
        corrupted_results['df']['Optimizer'] = optimizer_name
        corrupted_results['df']['Structure'] = structure
        corrupted_results['df']['k'] = k
        corrupted_results['df']['M'] = M
        corrupted_results['df']['gamma'] = gamma

        corrupted_results_df = pd.concat([corrupted_results_df, corrupted_results['df']], ignore_index=True)
    corrupted_results_df.rename(columns={'loss': 'NLL', 'accuracy': 'Accuracy', 'top_5_accuracy': 'Top-5 Accuracy',
                                         'ece': 'ECE', 'mce': 'MCE', 'uce': 'UCE', 'muce': 'MUCE', 'ace': 'ACE',
                                         'sce': 'SCE', 'brier': 'BS',
                                         'predictive_uncertainty': 'Predictive Uncertainty',
                                         'model_uncertainty': 'Model Uncertainty'},
                                inplace=True)
    return corrupted_results_df


def collect_corruption_errors(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    corruption_errors = pd.DataFrame()
    for run in runs:
        corruption_error = run['corrupted_results']['corruption_errors']
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


def collect_uncertainty(runs: Union[List[dict], dict]) -> pd.DataFrame:
    if type(runs) == dict:
        runs = [runs]
    df_uncertainty = pd.DataFrame(columns=['Dataset', 'Model', 'Method', 'Predictive Uncertainty',
                               'Model Uncertainty', 'severity', 'corruption type'])
    for run in runs:
        uncertainty = run['corrupted_results']['uncertainty']
        for severity, corruption_type in uncertainty['predictive_uncertainty'].keys():
            df = pd.DataFrame()
            df['Model Uncertainty'] = uncertainty['model_uncertainty'][(severity, corruption_type)]
            df['Predictive Uncertainty'] = uncertainty['predictive_uncertainty'][(severity, corruption_type)]
            df['severity'] = severity
            df['corruption type'] = corruption_type
            df['Method'] = run['method']
            df['Dataset'] = run['dataset']
            df['Model'] = run['model_name']
            df_uncertainty = pd.concat([df_uncertainty, df])
    return df_uncertainty


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
    bin_accuracies = result['r_counts'] * result['accuracies']
    bin_confidences = result['r_counts'] * result['confidences']
    r_bin_counts = result['r_counts']
    bin_errors = result['u_counts'] * result['errors']
    bin_uncertainties = result['u_counts'] * result['uncertainties']
    u_bin_counts = result['u_counts']

    for bin_data in data[1:]:
        assert(len(bins) == len(bin_data['bins']))
        bin_accuracies += bin_data['r_counts'] * bin_data['accuracies']
        bin_confidences += bin_data['r_counts'] * bin_data['confidences']
        r_bin_counts += bin_data['r_counts']

        bin_errors += bin_data['u_counts'] * bin_data['errors']
        bin_uncertainties += bin_data['u_counts'] * bin_data['uncertainties']
        u_bin_counts += bin_data['u_counts']

    bin_accuracies /= np.where(r_bin_counts > 0, r_bin_counts, 1)
    bin_confidences /= np.where(r_bin_counts > 0, r_bin_counts, 1)
    bin_errors /= np.where(u_bin_counts > 0, u_bin_counts, 1)
    bin_uncertainties /= np.where(u_bin_counts > 0, u_bin_counts, 1)

    avg_acc = np.sum(bin_accuracies * r_bin_counts) / np.sum(r_bin_counts)
    avg_conf = np.sum(bin_confidences * r_bin_counts) / np.sum(r_bin_counts)
    avg_err = np.sum(bin_errors * u_bin_counts) / np.sum(u_bin_counts)
    avg_uncert = np.sum(bin_uncertainties * u_bin_counts) / np.sum(u_bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * r_bin_counts) / np.sum(r_bin_counts)
    mce = np.max(gaps)

    gaps = np.abs(bin_errors - bin_uncertainties)
    uce = np.sum(gaps * u_bin_counts) / np.sum(u_bin_counts)
    muce = np.max(gaps)

    bin_data = dict(
        accuracies=bin_accuracies,
        confidences=bin_confidences,
        errors=bin_errors,
        uncertainties=bin_uncertainties,
        r_counts=r_bin_counts,
        u_counts=u_bin_counts,
        bins=bins,
        avg_accuracy=avg_acc,
        avg_confidence=avg_conf,
        avg_error=avg_err,
        avg_uncertainty=avg_uncert,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        expected_uncertainty_error=uce,
        max_uncertainty_error=muce,
    )
    return bin_data


def get_uncertainty(logits):
    return dict(
        model_uncertainty=model_uncertainty(logits).detach().cpu().numpy(),
        predictive_uncertainty=predictive_uncertainty(logits).detach().cpu().numpy(),
        # data_uncertainty=data_uncertainty(logits).detach().cpu().numpy()
    )


def get_bin_data(logits, labels, num_classes=-1, n_bins=10):
    if num_classes == -1:
        num_classes = F.one_hot(labels).shape[-1]
    probs = logits.softmax(-1)
    if len(probs.shape) == 3:
        probs = probs.mean(0)
    elif len(probs.shape) == 4:
        probs = probs.mean(1).mean(0)
    num_classes = torch.tensor(num_classes, dtype=float)
    uncertainties = 1/torch.log(num_classes) * predictive_uncertainty(logits)
    confidences, preds = probs.max(-1)

    uncertainties = uncertainties.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    bin_accuracies = np.zeros(n_bins, dtype=float)
    bin_confidences = np.zeros(n_bins, dtype=float)
    r_bin_counts = np.zeros(n_bins, dtype=int)
    bin_errors = np.zeros(n_bins, dtype=float)
    bin_uncertainties = np.zeros(n_bins, dtype=float)
    u_bin_counts = np.zeros(n_bins, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    r_indices = np.digitize(confidences, bins, right=True)
    u_indices = np.digitize(uncertainties, bins, right=True)

    for b in range(n_bins):
        selected = np.where(r_indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(labels[selected] == preds[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            r_bin_counts[b] = len(selected)
        selected = np.where(u_indices == b + 1)[0]
        if len(selected) > 0:
            bin_errors[b] = np.mean(labels[selected] != preds[selected])
            bin_uncertainties[b] = np.mean(uncertainties[selected])
            u_bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * r_bin_counts) / np.sum(r_bin_counts)
    avg_conf = np.sum(bin_confidences * r_bin_counts) / np.sum(r_bin_counts)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * r_bin_counts) / np.sum(r_bin_counts)
    mce = np.max(gaps)

    avg_err = np.sum(bin_errors * u_bin_counts) / np.sum(u_bin_counts)
    avg_uncert = np.sum(bin_uncertainties * u_bin_counts) / np.sum(u_bin_counts)
    gaps = np.abs(bin_errors - bin_uncertainties)
    uce = np.sum(gaps * u_bin_counts) / np.sum(u_bin_counts)
    muce = np.max(gaps)

    bin_data = dict(
        accuracies=bin_accuracies,
        confidences=bin_confidences,
        errors=bin_errors,
        uncertainties=bin_uncertainties,
        r_counts=r_bin_counts,
        u_counts=u_bin_counts,
        bins=bins,
        avg_accuracy=avg_acc,
        avg_confidence=avg_conf,
        avg_error=avg_err,
        avg_uncertainty=avg_uncert,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        expected_uncertainty_error=uce,
        max_uncertainty_error=muce
    )
    return bin_data


def make_csv(runs):
    if not os.path.exists('results'):
        os.mkdir('results')
    collect_results(runs).copy().to_csv('results/results.csv', index=False)
    collect_corrupted_results_df(runs).copy().to_csv('results/corrupted_results.csv', index=False)
    collect_corruption_errors(runs).copy().to_csv('results/corruption_errors.csv', index=False)
    # results_table().copy().to_csv('results/table.csv', index=True)


# def results_table():
#     corrupted_results = pd.read_csv('results/corrupted_results.csv')
#     corrupted_results_all = corrupted_results.copy()
#     corrupted_results_all['corruption_type'] = 'all'
#     corrupted_results = pd.concat([corrupted_results, corrupted_results_all])
#     corrupted_results = corrupted_results.groupby(['Dataset', 'Method', 'corruption_type']).agg(['mean', 'std']).fillna(0)
#
#     corruption_errors = pd.read_csv('results/corruption_errors.csv')
#     corruption_errors_all = corruption_errors.copy()
#     corruption_errors_all['Type'] = 'all'
#     corruption_errors = pd.concat([corruption_errors, corruption_errors_all])
#     corruption_errors = corruption_errors.groupby(['Dataset', 'Method', 'Type']).agg(['mean', 'std'])
#
#     # f_interval = lambda x: rf"{100 * x.mean():.1f} ($\pm$ {100 * x.std():.1f})"
#     # corrupted_results = corrupted_results[corrupted_results.corruption_type != 'clean']. \
#     #     drop(['severity', 'Loss', 'Top-5 Accuracy'], axis=1).groupby(['optimizer', 'corruption_type']). \
#     #     agg(f_interval).unstack('corruption_type').swaplevel(axis=1)
#     # corrupted_results_all = pd.read_csv('results/corrupted_results.csv')
#     #
#     # corrupted_results_all = corrupted_results_all[corrupted_results_all.corruption_type != 'clean']. \
#     #     drop(['severity', 'Loss', 'Top-5 Accuracy'], axis=1).groupby(['optimizer']).agg(f_interval)
#     # corrupted_results = pd.concat([corrupted_results, pd.concat({'all': corrupted_results_all}, axis=1)], axis=1)
#     #
#     # for c in CORRUPTION_TYPES.keys():
#     #     f_interval = lambda x: rf"{100 * x[c]:.1f} ($\pm$ {100 * x[f'{c}_std']:.1f})"
#     #     corrupted_results[c, 'mCE'] = corruption_errors[[c, f"{c}_std"]].apply(
#     #         f_interval, axis=1
#     #     )
#     #     corrupted_results[c, 'Rel. mCE'] = rel_corruption_errors[[c, f"{c}_std"]].apply(
#     #         f_interval, axis=1
#     #     )
#     # corrupted_results.sort_index(inplace=True)
#     # corrupted_results.rename(
#     #     columns={'all': 'All', 'noise': 'Noise', 'blur': 'Blur', 'weather': 'Weather', 'digital': 'Digital'}, inplace=True
#     # )
#     # corrupted_results = corrupted_results.reindex(
#     #     columns=corrupted_results.columns.reindex(['All', 'Noise', 'Blur', 'Weather', 'Digital'], level=0)[0]
#     # )
#     # corrupted_results = corrupted_results.reindex(
#     #     columns=corrupted_results.columns.reindex(['mCE', 'Rel. mCE', 'Accuracy', 'ECE', 'MCE'], level=1)[0]
#     # )
#     return corruption_errors, corrupted_results



def get_methods_and_model(dataset, model, model_params, optimizer, ngd_params=None, baseline='SGD',
                          use_cuda=torch.cuda.is_available()):
    device = 'cuda' if use_cuda else 'cpu'
    optimizer = optimizer.__name__
    if model == 'DeepEnsemble':
        runs = load_all_runs()
        runs = [run for run in runs if run['method'] == 'Vanilla' and run['dataset'].lower() == dataset.lower()]
        assert(len(runs) > 0)
        models = []
        for run in runs:
            state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt", map_location=device)['model_state_dict']
            model = Model(run['model_name'], **model_params)
            model.load_state_dict(state_dict=state_dict)
            models.append(model)
        model = DeepEnsemble(models=models, **model_params)
        methods = ['Deep Ensemble']
    elif model.startswith('NGDEnsemble'):
        # model of the form 'NGDDeepEnsemble|structure=<structure>;k=<value>'
        structure = model.split('|')[1].split(';')[0].split('=')[1]
        k = int(model.split('|')[1].split(';')[1].split('=')[1])
        runs = load_all_runs()
        # Load all NGD runs with same hyperparameters
        runs = [
            run for run in runs if run['method'].startswith('NGD')
                                   and run['params']['k'] == k
                                   and run['params']['structure'] in ['diagonal', structure]
                                   and run['params']['mc_samples'] == 1
                                   and run['params']['gamma'] == 1.0
                                   and run['dataset'].lower() == dataset.lower()
        ]
        assert(len(runs) > 0)
        structure = structure.replace('_', ' ').title().replace(' ', '')
        if k == 0:
            structure = 'Diagonal'
        method = rf"NGD Ensemble (structure = {structure}, $k = {k}$)"
        models = []
        optimizers = []
        for run in runs:
            state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt", map_location=device)

            model = Model(run['model_name'], **model_params)
            model.load_state_dict(state_dict=state_dict['model_state_dict'])
            models.append(model)

            optimizer = StructuredNGD(model.parameters(), state_dict['train_size'], **run['params'])
            optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
            optimizers.append(optimizer)
        model = HyperDeepEnsemble(models=models, optimizers=optimizers, **model_params)
        methods = [method]
    elif model.startswith('HyperDeepEnsemble'):
        # model of the form 'HyperDeepEnsemble|structure=<structure>;<parameter>=<value>'
        method = 'Hyper-Deep Ensemble'
        structure = model.split('|')[1].split(';')[0].split('=')[1]
        param, value = model.split('|')[1].split(';')[1].split('=')
        value = eval(value)
        assert(param in ['gamma', 'M', 'mc_samples'])
        # Number of MC samples in parameter is called 'mc_samples'
        if param in ['M', 'mc_samples']:
            value = int(value)
            other_param = 'gamma'
            method = rf"{method} (structure = {structure.replace('_', ' ').title().replace(' ', '')}, $M = {value}$)"
        else:
            other_param = 'mc_samples'
            method = rf"{method} (structure = {structure.replace('_', ' ').title().replace(' ', '')}, $\gamma = {value}$)"
        if param == 'M':
            param = 'mc_samples'
        # Load all runs (not only the unique ones) and take NGD runs with the same parameter value as specified
        # Keep all model that have the given value and keep only the specified structure plus the diagonal model
        # (whose structure is called 'diagonal', differently from the specified structures)
        runs = load_all_runs(method_unique=False)
        runs = [run for run in runs if run['method'].startswith('NGD') and run['dataset'].lower() == dataset.lower()]
        runs = [run for run in runs if run['params'][param] == value and run['params'][other_param] == 1
                and (run['params']['structure'] in ['diagonal', structure])]
        assert(len(runs) > 0)
        models = []
        optimizers = []
        for run in runs:
            state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt", map_location=device)

            model = Model(run['model_name'], **model_params)
            model.load_state_dict(state_dict=state_dict['model_state_dict'])
            models.append(model)

            optimizer = StructuredNGD(model.parameters(), state_dict['train_size'], **run['params'])
            optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])
            optimizers.append(optimizer)
        model = HyperDeepEnsemble(models=models, optimizers=optimizers, **model_params)
        methods = [method]
    elif model == 'BBB':
        runs = load_all_runs()
        run = [run for run in runs if run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], bnn=True, **model_params)
        methods = ['BBB']
    elif model == 'Dropout':
        runs = load_all_runs()
        run = [run for run in runs if run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], dropout_layers='all', p=0.2, **model_params)
        methods = ['Dropout']
    elif model == 'LLDropout':
        runs = load_all_runs()
        run = [run for run in runs if run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        model = Model(model_type=run['model_name'], dropout_layers='last', p=0.2, **model_params)
        methods = ['LL Dropout']
    elif model == 'TempScaling':
        runs = load_all_runs()
        run = [run for run in runs if run['method'] == 'Vanilla'
               and run['dataset'].lower() == dataset.lower()][0]
        state_dict = torch.load(f"checkpoints/{run['timestamp']}.pt", map_location=device)['model_state_dict']
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
                if param['k'] == 0:
                    structure = 'Diagonal'
                method = rf"NGD (structure = {structure}, $k = {param['k']})$"
                methods.append(method)
    return methods, model


class TinyImageNet(ImageFolder):
    def __init__(self, root='/storage/group/dataset_mirrors/old_common_datasets/tiny-imagenet-200/',
                 split='train',
                 transform=transforms.Compose([transforms.ToTensor()])):
        super().__init__(os.path.join(root, split), transform)
        self.root = '/ImageNet'
