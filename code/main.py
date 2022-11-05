from network import Model
from plot_runs import *
from metrics import *
from models import *


def main() -> None:
	"""Run script.
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	args = parse_args()

	mce = MCE(args['n_bins'])
	ece = ECE(args['n_bins'])
	top_k_accuracy = TopkAccuracy(top_k=5)
	metrics = [accuracy, top_k_accuracy, ece, mce]  # precision, recall, f1_score,

	train_loader, val_loader, test_loader = load_data(args['dataset'], args['batch_size'], args['data_split'])
	num_classes = len(train_loader.dataset.classes)
	n = len(train_loader.dataset)
	input_shape = iter(train_loader).next()[0].shape[1:]

	if args['model'] == 'DeepEnsemble':
		runs = load_all_runs()
		runs = [run for run in runs if run['optimizer_name'] == args['baseline']
				and run['dataset'].lower() == args['dataset'].lower()]
		model = DeepEnsemble(runs=runs, num_classes=num_classes, device=device, input_shape=input_shape)
		args['optimizers'] = [eval(args['baseline'])]
	elif args['model'] == 'TempScaling':
		runs = load_all_runs()
		runs = [x for x in runs if x['optimizer_name'] == args['baseline']
				and x['dataset'].lower() == args['dataset'].lower()]
		run = runs[0]
		model = Model(model_type=run['model_name'], num_classes=num_classes,
					  device=device, input_shape=input_shape)
		model = TempScaling(model)
		args['optimizers'] = [eval(args['baseline'])]
	else:
		model = Model(model_type=args['model'], num_classes=num_classes,
					  device=device, input_shape=input_shape)

	params = get_params(args, baseline=args['baseline'], n=n)
	runs = run_experiments(
		args['epochs'], model, args['optimizers'], train_loader, val_loader, test_loader, args['baseline'],
		baseline_params=params['baseline'], ngd_params=params['ngd'], metrics=metrics,
		eval_every=args['eval_every'], n_bins=args['n_bins'], mc_samples=args['mc_samples_eval']
	)
	save_runs(runs)


if __name__ == '__main__':
	main()
