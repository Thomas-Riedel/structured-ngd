from util import *


def get_corrupted_results(dataset, model, optimizer, method, baseline, metrics, clean_results, mc_samples, n_bins):
    if not dataset.lower() in ['cifar10', 'cifar100']:
        return None
    optimizer_name = optimizer.__name__ if isinstance(optimizer, NoisyOptimizer) else type(optimizer).__name__
    model_name = model.__name__

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
    corruption_errors = pd.DataFrame()
    bin_data = {(severity, corruption_type): clean_results['bin_data']}
    uncertainty = dict(
        model_uncertainty={(severity, corruption_type): clean_results['uncertainty']['model_uncertainty']},
        predictive_uncertainty={(severity, corruption_type): clean_results['uncertainty']['predictive_uncertainty']}
    )
    i = 1
    for severity in SEVERITY_LEVELS:
        for corruption_type in CORRUPTION_TYPES.keys():
            if corruption_type == 'all':
                continue
            bin_data_list = []
            for corruption in CORRUPTION_TYPES[corruption_type]:
                print('----------------------------------------------------------------')
                print(f"[{i} / {len(SEVERITY_LEVELS) * len(CORRUPTIONS)}]; "
                      f"severity = {severity}, corruption = {corruption}\n")
                data_loader = load_corrupted_data(dataset, corruption, severity)
                loss, metric, corrupted_bin_data, corrupted_uncertainty = model.evaluate(
                    data_loader, metrics=metrics, optimizer=optimizer, mc_samples=mc_samples, n_bins=n_bins
                )
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

            # group bin_data results by types
            corrupted_bin_data = merge_bin_data(bin_data_list)
            bin_data[(severity, corruption_type)] = corrupted_bin_data
            uncertainty['model_uncertainty'][(severity, corruption_type)] = corrupted_uncertainty['model_uncertainty']
            uncertainty['predictive_uncertainty'][(severity, corruption_type)] = corrupted_uncertainty['predictive_uncertainty']

    sub_df = df[
        (df['severity'] > 0) & (df['corruption_type'] != 'all')
    ].drop(['corruption_type', 'severity'], axis=1).copy()
    clean_accuracy = clean_results['test_metrics']['accuracy']
    if isinstance(optimizer, NoisyOptimizer) or method != 'Vanilla':
        baseline_results = load_run(dataset, model, baseline, method='Vanilla')
        baseline_clean_accuracy = baseline_results['test_metrics']['accuracy']
        baseline_df = baseline_results['corrupted_results']['df']
        baseline_df = baseline_df[
            (baseline_df['severity'] > 0) & (df['corruption_type'] != 'all')
        ].drop(['corruption_type', 'severity'], axis=1).copy()
    else:
        baseline_clean_accuracy = clean_accuracy
        baseline_df = sub_df.copy()
    corruption_errors['mCE'] = ce(sub_df, baseline_df).reindex(columns=CORRUPTIONS).values.reshape(-1)
    corruption_errors['Rel. mCE'] = rel_ce(sub_df, baseline_df, clean_accuracy, baseline_clean_accuracy).reindex(columns=CORRUPTIONS).values.reshape(-1)
    corruption_errors['Method'] = method
    corruption_errors['Model'] = model_name
    corruption_errors['Dataset'] = dataset
    corruption_errors['Optimizer'] = optimizer_name
    corruption_errors['Corruption'] = CORRUPTIONS
    corruption_errors['Type'] = corruption_errors['Corruption'].apply(get_corruption_type)
    corruption_errors['Accuracy'] = clean_results['test_metrics']['accuracy']

    df['Method'] = method
    df['Dataset'] = dataset
    df['Model'] = model_name
    df['Optimizer'] = optimizer_name

    corrupted_results = dict(
        model_name=model_name,
        optimizer_name=optimizer_name,
        method=method,
        dataset=dataset,
        corruption_errors=corruption_errors,
        df=df,
        bin_data=bin_data,
        uncertainty=uncertainty
    )
    return corrupted_results


def get_corruption_type(x):
    if x in ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise']:
        return 'noise'
    if x in ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur']:
        return 'blur'
    if x in ['snow', 'frost', 'fog', 'brightness', 'spatter']:
        return 'weather'
    if x in ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate']:
        return 'digital'
    return 'all'


def ce(df, baseline_corrupted_df):
    result = pd.DataFrame()
    for corruption in CORRUPTIONS:
        sub_df = df[df.corruption == corruption].copy()
        sub_baseline_df = baseline_corrupted_df[baseline_corrupted_df.corruption == corruption].copy()
        result[corruption] = [np.sum(1 - sub_df.accuracy) / np.sum(1 - sub_baseline_df.accuracy)]
    return result


def rel_ce(df, baseline_corrupted_df, clean_accuracy, baseline_clean_accuracy):
    result = pd.DataFrame()
    for corruption in CORRUPTIONS:
        sub_df = df[df.corruption == corruption].copy()
        sub_baseline_df = baseline_corrupted_df[baseline_corrupted_df.corruption == corruption].copy()
        result[corruption] = [np.sum(clean_accuracy - sub_df.accuracy) / np.sum(baseline_clean_accuracy - sub_baseline_df.accuracy)]
    return result

