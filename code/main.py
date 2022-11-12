from plot_runs import *
from metrics import *


def main() -> None:
	"""Run script.
	"""
	args = parse_args()

	device = 'cuda' if args['use_cuda'] else 'cpu'
	train_loader, val_loader, test_loader = load_data(args['dataset'], args['batch_size'], args['data_split'])
	num_classes = len(train_loader.dataset.dataset.classes)
	n = len(train_loader.dataset)
	input_shape = iter(train_loader).next()[0].shape[1:]

	accuracy = Accuracy()
	top_k_accuracy = TopkAccuracy(top_k=5)
	ece = ECE(args['n_bins'])
	uce = UCE(num_classes, args['n_bins'])
	mce = MCE(args['n_bins'])
	muce = MUCE(num_classes, args['n_bins'])
	ace = ACE()
	sce = SCE()
	brier = Brier(num_classes)
	model_uncert = ModelUncertainty()
	pred_uncert = PredictiveUncertainty()

	metrics = [accuracy, ece, uce, mce, muce, top_k_accuracy, sce, ace, brier, model_uncert, pred_uncert]

	params = get_params(args, optimizer=args['optimizer'], n=n)
	model_params = dict(num_classes=num_classes, input_shape=input_shape, device=device)
	methods, model = get_methods_and_model(
		args['dataset'], args['model'],  model_params, args['optimizer'], params['ngd'], args['baseline'], args['use_cuda']
	)

	runs = run_experiments(
		args['epochs'], methods, model, args['optimizer'], train_loader, val_loader, test_loader, args['baseline'],
		baseline_params=params['baseline'], ngd_params=params['ngd'], metrics=metrics, eval_every=args['eval_every'],
		n_bins=args['n_bins'], mc_samples_val=args['mc_samples_val'], mc_samples_test=args['mc_samples_test']
	)
	save_runs(runs)


if __name__ == '__main__':
	main()
