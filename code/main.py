from torchmetrics.functional import accuracy, precision, recall, f1_score, calibration_error
from models.resnet import Model
from plot_runs import *


class ECE:
	def __init__(self, n_bins=10):
		self.n_bins = n_bins
		self.__name__ = 'ece'

	def __call__(self, logits, labels):
		return calibration_error(logits, labels, n_bins=self.n_bins, norm='l1')


class MCE:
	def __init__(self, n_bins=10):
		self.n_bins = n_bins
		self.__name__ = 'mce'

	def __call__(self, logits, labels):
		return calibration_error(logits, labels, n_bins=self.n_bins, norm='max')


def main() -> None:
	"""Run script.
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	args = parse_args()

	mce = MCE(args['n_bins'])
	ece = ECE(args['n_bins'])
	metrics = [accuracy, precision, recall, f1_score, ece, mce]

	train_loader, val_loader, test_loader = load_data(args['dataset'], args['batch_size'], args['data_split'])
	num_classes = len(train_loader.dataset.classes)
	n = len(train_loader.dataset)
	input_shape = iter(train_loader).next()[0].shape[1:]
	model = Model(model_type=args['model'], num_classes=num_classes, device=device, input_shape=input_shape)

	params = get_params(args, n=n)
	runs = run(
		args['epochs'], model, args['optimizers'], train_loader, val_loader, test_loader, adam_params=params['adam'],
		ngd_params=params['ngd'], metrics=metrics, eval_every=args['eval_every'], n_bins=args['n_bins']
	)
	save_runs(runs)


if __name__ == '__main__':
	main()
