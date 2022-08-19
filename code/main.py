import argparse
import sys
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle

import torch
import torch.nn as nn

from torchvision.datasets import FashionMNIST, MNIST, CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from torchmetrics.functional import calibration_error
from torch.optim import Adam

from models.resnet import *
from optimizers.rank_k_cov import *


def parse_args():
	parser = argparse.ArgumentParser(description='Run noisy optimizers with parameters.')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--dataset', type=str, default="CIFAR10")
	parser.add_argument('--model', type=str, default="resnet18")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--mc_samples', type=int, default=1)
	parser.add_argument('--eval_every', type=int, default=10)

	args = parser.parse_args(sys.argv[1:])
	return args.epochs, args.dataset, args.model, args.batch_size, args.mc_samples, args.eval_every


def load_data(dataset, batch_size):
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


def run(epochs, model, optimizers, train_loader, val_loader, mc_samples=1, eval_every=1):
	loss_fn = nn.CrossEntropyLoss()
	runs = []
	lr = 1e-2
	device = model.device

	for optimizer in optimizers:
		model.init_weights()
		if optimizer is RankCov:
			optimizer = optimizer(model.parameters(), len(train_loader.dataset), rank=0, lr=lr, device=device, mc_samples=mc_samples)
		else:
			optimizer = optimizer(model.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

		times = []
		val_acc = []
		val_loss = []
		iter_losses = []

		for epoch in range(epochs):
			start = time.time()
			running_loss, iter_loss = model.train(train_loader, optimizer, epoch=epoch,
												  loss_fn=loss_fn, eval_every=eval_every)
			if epoch == 0:
				times.append(0)
			else:
				times.append(time.time() - start)

			acc, loss = model.evaluate(val_loader)
			val_acc.append(acc)
			val_loss.append(loss)
			iter_losses += iter_loss
			scheduler.step()

		runs.append(
			dict(
				name=type(optimizer).__name__,
				optimizer=optimizer,
				times=np.cumsum(times),
				val_acc=val_acc,
				val_loss=val_loss,
				iter_loss=iter_losses,
				time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
		print('Finished Training')
	return runs


def plot_runs(runs):
	# Plot loss per iterations
	plt.figure(figsize=(12, 8))
	for run in runs:
		plt.plot(run['iter_loss'], label=run['name'])
		plt.ylim(bottom=0)
		plt.title('Training Curve')
		plt.xlabel('Iterations')
		plt.legend()
		plt.savefig(f"plots/{run['time']}_iter")
		plt.show()

	# Plot loss and accuracy over epochs and time
	plt.figure(figsize=(12, 8))
	for run in runs:
		plt.subplot(2, 2, 1)
		plt.plot(run['val_acc'], label=run['name'])
		plt.title('Validation Accuracy')
		plt.xlabel('epochs')
		plt.ylim(0, 1)
		plt.legend()

		plt.subplot(2, 2, 2)
		plt.plot(run['val_loss'], label=run['name'])
		plt.title('Validation Loss')
		plt.xlabel('epochs')
		plt.ylim(bottom=0)
		plt.legend()

		plt.subplot(2, 2, 3)
		plt.plot(run['times'], run['val_acc'], label=run['name'])
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
		plt.savefig(f"plots/{run['time']}_loss_acc")
		plt.show()


def save_runs(runs):
	for run in runs:
		with open(f"runs/{run['time']}.pkl", 'wb') as f:
			pickle.dump(run, f)


def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	EPOCHS, DATASET, MODEL, BATCH_SIZE, MC_SAMPLES, EVAL_EVERY = parse_args()

	train_loader, val_loader, test_loader = load_data(DATASET, BATCH_SIZE)
	num_classes = len(train_loader.dataset.classes)

	model = ResNet(model_type=MODEL, num_classes=num_classes, device=device)
	optimizers = [RankCov]

	runs = run(EPOCHS, model, optimizers, train_loader, val_loader, mc_samples=MC_SAMPLES, eval_every=EVAL_EVERY)
	plot_runs(runs)
	save_runs(runs)


if __name__ == '__main__':
	main()
