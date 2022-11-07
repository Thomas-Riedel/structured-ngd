from plot_runs import *
from metrics import *


def main() -> None:
	"""Run script.
	"""
	args = parse_args()

	device = 'cuda' if args['use_cuda'] else 'cpu'
	train_loader, val_loader, test_loader = load_data(args['dataset'], args['batch_size'], args['data_split'])
	num_classes = len(train_loader.dataset.classes)
	n = len(train_loader.dataset)
	input_shape = iter(train_loader).next()[0].shape[1:]

	ece = ECE(args['n_bins'])
	uce = UCE(num_classes, args['n_bins'])
	mce = MCE(args['n_bins'])
	muce = MUCE(num_classes, args['n_bins'])
	top_k_accuracy = TopkAccuracy(top_k=5)
	top_k_ece = TopkECE(top_k=5, n_bins=args['n_bins'])
	top_k_uce = TopkUCE(num_classes=num_classes, top_k=5, n_bins=args['n_bins'])
	top_k_mce = TopkMCE(top_k=5, n_bins=args['n_bins'])
	top_k_muce = TopkMUCE(num_classes=num_classes, top_k=5, n_bins=args['n_bins'])
	brier = Brier(num_classes)
	metrics = [accuracy, ece, uce, mce, muce, brier, top_k_accuracy, top_k_ece, top_k_uce, top_k_mce, top_k_muce]

	params = get_params(args, baseline=args['baseline'], n=n)
	model_params = dict(num_classes=num_classes, input_shape=input_shape, device=device)
	methods, model = get_methods_and_model(
		args['dataset'], args['model'],  model_params, args['optimizer'], params['ngd'], args['baseline']
	)

	runs = run_experiments(
		args['epochs'], methods, model, args['optimizer'], train_loader, val_loader, test_loader, args['baseline'],
		baseline_params=params['baseline'], ngd_params=params['ngd'], metrics=metrics,
		eval_every=args['eval_every'], n_bins=args['n_bins'], mc_samples=args['mc_samples_eval']
	)
	save_runs(runs)


if __name__ == '__main__':
	main()
