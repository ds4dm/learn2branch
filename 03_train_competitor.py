import pickle
import os
import argparse
import numpy as np
import utilities
import pathlib

from utilities import log, load_flat_samples


def load_samples(filenames, feat_type, label_type, augment, qbnorm, size_limit, logfile=None):
    x, y, ncands = [], [], []
    total_ncands = 0

    for i, filename in enumerate(filenames):
        cand_x, cand_y, best = load_flat_samples(filename, feat_type, label_type, augment, qbnorm)

        x.append(cand_x)
        y.append(cand_y)
        ncands.append(cand_x.shape[0])
        total_ncands += ncands[-1]

        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{len(filenames)} files processed ({total_ncands} candidate variables)", logfile)

        if total_ncands >= size_limit:
            log(f"  dataset size limit reached ({size_limit} candidate variables)", logfile)
            break

    x = np.concatenate(x)
    y = np.concatenate(y)
    ncands = np.asarray(ncands)

    if total_ncands > size_limit:
        x = x[:size_limit]
        y = y[:size_limit]
        ncands[-1] -= total_ncands - size_limit

    return x, y, ncands


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='Model to be trained.',
        type=str,
        choices=['svmrank', 'extratrees', 'lambdamart'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    args = parser.parse_args()

    feats_type = 'nbr_maxminmean'

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
    }
    problem_folder = problem_folders[args.problem]

    if args.model == 'extratrees':
        train_max_size = 250000
        valid_max_size = 100000
        feat_type = 'gcnn_agg'
        feat_qbnorm = False
        feat_augment = False
        label_type = 'scores'

    elif args.model == 'lambdamart':
        train_max_size = 250000
        valid_max_size = 100000
        feat_type = 'khalil'
        feat_qbnorm = True
        feat_augment = False
        label_type = 'bipartite_ranks'

    elif args.model == 'svmrank':
        train_max_size = 250000
        valid_max_size = 100000
        feat_type = 'khalil'
        feat_qbnorm = True
        feat_augment = True
        label_type = 'bipartite_ranks'

    rng = np.random.RandomState(args.seed)

    running_dir = f"trained_models/{args.problem}/{args.model}_{feat_type}/{args.seed}"
    os.makedirs(running_dir)

    logfile = f"{running_dir}/log.txt"
    log(f"Logfile for {args.model} model on {args.problem} with seed {args.seed}", logfile)

    # Data loading
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))

    log(f"{len(train_files)} training files", logfile)
    log(f"{len(valid_files)} validation files", logfile)

    log("Loading training samples", logfile)
    train_x, train_y, train_ncands = load_samples(
            rng.permutation(train_files),
            feat_type, label_type, feat_augment, feat_qbnorm,
            train_max_size, logfile)
    log(f"  {train_x.shape[0]} training samples", logfile)

    log("Loading validation samples", logfile)
    valid_x, valid_y, valid_ncands = load_samples(
            valid_files,
            feat_type, label_type, feat_augment, feat_qbnorm,
            valid_max_size, logfile)
    log(f"  {valid_x.shape[0]} validation samples", logfile)

    # Data normalization
    log("Normalizing datasets", logfile)
    x_shift = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale == 0] = 1

    valid_x = (valid_x - x_shift) / x_scale
    train_x = (train_x - x_shift) / x_scale

    # Saving feature parameters
    with open(f"{running_dir}/feat_specs.pkl", "wb") as file:
        pickle.dump({
                'type': feat_type,
                'augment': feat_augment,
                'qbnorm': feat_qbnorm,
            }, file)

    # save normalization parameters
    with open(f"{running_dir}/normalization.pkl", "wb") as f:
        pickle.dump((x_shift, x_scale), f)

    log("Starting training", logfile)
    if args.model == 'extratrees':
        from sklearn.ensemble import ExtraTreesRegressor

        # Training
        model = ExtraTreesRegressor(
            n_estimators=100,
            random_state=rng,)
        model.verbose = True
        model.fit(train_x, train_y)
        model.verbose = False

        # Saving model
        with open(f"{running_dir}/model.pkl", "wb") as file:
            pickle.dump(model, file)

        # Testing
        loss = np.mean((model.predict(valid_x) - valid_y) ** 2)
        log(f"Validation RMSE: {np.sqrt(loss):.2f}", logfile)

    elif args.model == 'lambdamart':
        import pyltr

        train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
        valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands)

        # Training
        model = pyltr.models.LambdaMART(verbose=1, random_state=rng, n_estimators=500)
        model.fit(train_x, train_y, train_qids,
            monitor=pyltr.models.monitors.ValidationMonitor(
                valid_x, valid_y, valid_qids, metric=model.metric))

        # Saving model
        with open(f"{running_dir}/model.pkl", "wb") as file:
            pickle.dump(model, file)

        # Testing
        loss = model.metric.calc_mean(valid_qids, valid_y, model.predict(valid_x))
        log(f"Validation log-NDCG: {np.log(loss)}", logfile)

    elif args.model == 'svmrank':
        import svmrank

        train_qids = np.repeat(np.arange(len(train_ncands)), train_ncands)
        valid_qids = np.repeat(np.arange(len(valid_ncands)), valid_ncands)

        # Training (includes hyper-parameter tuning)
        best_loss = np.inf
        best_model = None
        for c in (1e-3, 1e-2, 1e-1, 1e0):
            log(f"C: {c}", logfile)
            model = svmrank.Model({
                '-c': c * len(train_ncands),  # c_light = c_rank / n
                '-v': 1,
                '-y': 0,
                '-l': 2,
            })
            model.fit(train_x, train_y, train_qids)
            loss = model.loss(train_y, model(train_x, train_qids), train_qids)
            log(f"  training loss: {loss}", logfile)
            loss = model.loss(valid_y, model(valid_x, valid_qids), valid_qids)
            log(f"  validation loss: {loss}", logfile)
            if loss < best_loss:
                best_model = model
                best_loss = loss
                best_c = c
                # save model
                model.write(f"{running_dir}/model.txt")

        log(f"Best model with C={best_c}, validation loss: {best_loss}", logfile)
