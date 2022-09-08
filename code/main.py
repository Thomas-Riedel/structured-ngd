import argparse
import sys
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle
import re
import itertools
from typing import Union, Tuple, List, Callable

from torchvision.datasets import FashionMNIST, MNIST, CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from torchmetrics.functional import accuracy, precision, recall, f1_score, calibration_error
from torch.optim import Adam

from models.resnet import *
from optimizers.rank_k_cov import *


def parse_args() -> Tuple[int, str, str, int, List[float], List[int], List[int], str, int]:
	parser = argparse.ArgumentParser(description='Run noisy optimizers with parameters.')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--dataset', type=str, default="CIFAR10")
	parser.add_argument('--model', type=str, default="resnet18")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--lr', type=str, default='1e-3')
	parser.add_argument('--k', type=str, default='0')
	parser.add_argument('--mc_samples', type=str, default='1')
	parser.add_argument('--structure', type=str, default='rank_cov')
	parser.add_argument('--eval_every', type=int, default=10)

	args = parser.parse_args(sys.argv[1:])
	args.lr, args.k, args.mc_samples = parse_vals([args.lr, args.k, args.mc_samples],
												  [float, int, int])
	args.structure = len(args.lr) * [args.structure]

	return args.epochs, args.dataset, args.model, args.batch_size, args.lr, args.k, args.mc_samples, args.structure, args.eval_every


def parse_vals(args: List[str], types: List[type]) -> List[Union[int, float]]:
	def make_sequence(type):
		def f(val):
			if type == int:
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


def load_data(dataset: str, batch_size: int) -> Tuple[DataLoader]:
	"""Load dataset, prepare for ResNet training, and split into train. validation and test set
	"""

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
	split = int(0.8 * len(training_data))
	train_sampler = SubsetRandomSampler(indices[:split])
	val_sampler = SubsetRandomSampler(indices[split:])

	train_loader = DataLoader(training_data, sampler=train_sampler, batch_size=batch_size)
	val_loader = DataLoader(training_data, sampler=val_sampler, batch_size=batch_size)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_loader, val_loader, test_loader


def run(epochs: int, model: str, optimizers: List[Union[StructuredNGD, Adam]],
		train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None,
		adam_params: List[dict] = None, ngd_params: List[dict] = None, metrics: List[Callable] = [],
		eval_every: int = 1) -> List[dict]:
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

			times = []
			val_loss = []
			val_metrics = {}
			iter_losses = []
			iter_metrics = {}
			for metric in metrics:
				val_metrics[metric.__name__] = []
				iter_metrics[metric.__name__] = []

			for epoch in range(epochs):
				start = time.time()
				iter_loss, iter_metric = model.train(train_loader, optimizer, epoch=epoch,
													 loss_fn=loss_fn, metrics=metrics,
													 eval_every=eval_every)
				if epoch == 0:
					times.append(0)
				else:
					times.append(time.time() - start)

				loss, metric = model.evaluate(val_loader, metrics=metrics)
				# Append single validation metric value epoch-wise
				for metric_key in metric.keys():
					val_metrics[metric_key].append(metric[metric_key])
				val_loss.append(loss)

				iter_losses += iter_loss
				# Append multiple values iteration-wise
				for metric_key in iter_metric.keys():
					iter_metrics[metric_key] += iter_metric[metric_key]

				scheduler.step()

			test_metrics, test_loss = model.evaluate(test_loader, metrics=metrics)

			runs.append(
				dict(
					name=type(optimizer).__name__,
					optimizer=optimizer,
					params=params,
					times=np.cumsum(times),
					val_metrics=val_metrics,
					val_loss=val_loss,
					test_loss = test_loss,
					test_metrics=test_metrics,
					iter_loss=iter_losses,
					iter_metrics=iter_metrics,
					time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
		print('Finished Training')
	return runs


def plot_runs(runs: Union[dict, List[dict]]) -> None:
	if type(runs) == dict:
		runs = [runs]
	# Plot loss per iterations
	plt.figure(figsize=(12, 8))
	for run in runs:
		plt.plot(run['iter_loss'], label=run['name'])
		plt.ylim(bottom=0)
		plt.title('Training Curve')
		plt.xlabel('iterations')
		plt.legend()
		plt.savefig(f"plots/{run['time']}_iter_loss")
		plt.show()

		for metric_key in run['iter_metrics'].keys():
			plt.plot(run['iter_metrics'][metric_key],
					 label=f"{run['name']} ({metric_key})")

		plt.ylim(bottom=0)
		plt.title('Training Metrics')
		plt.xlabel('iterations')
		plt.legend()
		plt.savefig(f"plots/{run['time']}_iter_metrics")
		plt.show()

	# Plot loss and accuracy over epochs and time
	for run in runs:
		plt.figure(figsize=(12, 8))
		plt.subplot(2, 2, 1)
		plt.plot(run['val_loss'], label=run['name'])
		plt.title('Validation Loss')
		plt.xlabel('epochs')
		plt.ylim(bottom=0)
		plt.legend()

		plt.subplot(2, 2, 2)
		for metric_key in run['val_metrics'].keys():
			plt.plot(run['val_metrics'][metric_key],
					 label=f"{run['name']} ({metric_key})")

		plt.title('Validation Metrics')
		plt.xlabel('epochs')
		plt.ylim(0, 1)
		plt.legend()

		plt.subplot(2, 2, 3)
		for metric_key in run['val_metrics'].keys():
			plt.plot(run['times'], run['val_metrics'][metric_key],
					 label=f"{run['name']} ({metric_key})")
		plt.title('Validation Accuracy')
		plt.xlabel('time (s)')
		plt.ylim(0, 1)
		plt.legend()

		plt.subplot(2, 2, 4)
		plt.plot(run['times'], run['val_loss'], label=run['name'])
		plt.title('Validation Loss')
		plt.xlabel('time (s)')
		plt.ylim(bottom=0)
		plt.legend()

		plt.tight_layout()
		plt.savefig(f"plots/{run['time']}_loss_metrics")
		plt.show()


def save_runs(runs: Union[dict, List[dict]]) -> None:
	if type(runs) == dict:
		runs = [runs]
	for run in runs:
		with open(f"runs/{run['time']}.pkl", 'wb') as f:
			pickle.dump(run, f)


def main() -> None:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	EPOCHS, DATASET, MODEL, BATCH_SIZE, LEARNING_RATE, K, MC_SAMPLES, STRUCTURE, EVAL_EVERY = parse_args()
	metrics = [accuracy, precision, recall, f1_score, calibration_error]

	train_loader, val_loader, test_loader = load_data(DATASET, BATCH_SIZE)
	num_classes = len(train_loader.dataset.classes)

	model = ResNet(model_type=MODEL, num_classes=num_classes, device=device)
	ngd_params = [
		dict(lr=lr, k=k, mc_samples=mc_samples, structure=structure)
		for lr, k, mc_samples, structure in zip(LEARNING_RATE, K, MC_SAMPLES, STRUCTURE)
	]
	adam_params = [dict(lr=lr) for lr in set(LEARNING_RATE)]
	optimizers = [Adam, StructuredNGD]

	runs = run(
		EPOCHS, model, optimizers, train_loader, val_loader, adam_params=adam_params,
		ngd_params=ngd_params, metrics=metrics, eval_every=EVAL_EVERY
	)
	save_runs(runs)
	plot_runs(runs)


if __name__ == '__main__':
	main()
