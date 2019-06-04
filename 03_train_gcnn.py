import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utilities
from utilities import log


def load_batch(sample_files):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []
    cand_actionss = []
    expert_actions = []

    # load samples
    for filename in sample_files:
        with gzip.open(filename, 'rb') as f:
            sample = pickle.load(f)

        state, khalil_state, expert_action, cand_actions, *_ = sample['data']

        cand_actions = np.array(cand_actions)
        expert_action = np.where(cand_actions == expert_action)[0][0]  # best action relative to candidates

        c, e, v = state
        c_features.append(c['values'])
        e_indices.append(e['indices'])
        e_features.append(e['values'])
        v_features.append(v['values'])
        expert_actions.append(expert_action)
        cand_actionss.append(cand_actions)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cands.shape[0] for cands in cand_actionss]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)
    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
                                for j, e_ind in enumerate(e_indices)], axis=1)
    # candidate indexes as well
    cand_actionss = np.concatenate([cands + shift
        for cands, shift in zip(cand_actionss, cv_shift[1])])
    expert_actions = np.array(expert_actions)

    # convert to tensors
    c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)
    expert_actions = tf.convert_to_tensor(expert_actions, dtype=tf.int32)
    cand_actionss = tf.convert_to_tensor(cand_actionss, dtype=tf.int32)
    n_cands_per_sample = tf.convert_to_tensor(n_cands_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, cand_actionss, expert_actions


def load_batch_tf(x):
    return tf.py_func(
        load_batch,
        [x],
        [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])


def pretrain(model, dataloader):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : tf.data.Dataset
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        for batch in dataloader:
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, actions = batch
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            if not model.pre_train(batched_states):
                break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer, name = res

        i += 1

    return i


def process(model, dataloader, top_k, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, actions = batch
        batched_states = (c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True))  # prevent padding
        batch_size = len(n_cs.numpy())

        if optimizer:
            with tf.GradientTape() as tape:
                logits = model(batched_states)
                logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
                loss = tf.losses.sparse_softmax_cross_entropy(labels=actions, logits=logits)
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
        else:
            logits = model(batched_states)
            logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
            loss = tf.losses.sparse_softmax_cross_entropy(labels=actions, logits=logits)

        kacc = np.array([
            tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=actions, k=k), tf.float32)).numpy()
            for k in top_k
        ])
        mean_loss += loss.numpy() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities'],
    )
    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='baseline',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]

    if args.problem == 'setcover':
        train_files = list(pathlib.Path('data/samples/setcover/500r_1000c_0.05d/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path('data/samples/setcover/500r_1000c_0.05d/valid').glob('sample_*.pkl'))

    elif args.problem == 'cauctions':
        train_files = list(pathlib.Path('data/samples/cauctions/100_500/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path('data/samples/cauctions/100_500/valid').glob('sample_*.pkl'))

    elif args.problem == 'facilities':
        train_files = list(pathlib.Path('data/samples/facilities/100_100_5/train').glob('sample_*.pkl'))
        valid_files = list(pathlib.Path('data/samples/facilities/100_100_5/valid').glob('sample_*.pkl'))

    else:
        raise NotImplementedError

    running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"

    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### NUMPY / TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_data = valid_data.batch(valid_batch_size)
    valid_data = valid_data.map(load_batch_tf)
    valid_data = valid_data.prefetch(1)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
    pretrain_data = pretrain_data.batch(pretrain_batch_size)
    pretrain_data = pretrain_data.map(load_batch_tf)
    pretrain_data = pretrain_data.prefetch(1)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.GCNPolicy()
    del sys.path[0]

    ### TRAINING LOOP ###
    optimizer = tf.train.AdamOptimizer(learning_rate=lambda: lr)  # dynamic LR trick
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
            # model compilation
            model.call = tfe.defun(model.call, input_signature=model.input_signature)
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
            train_data = train_data.batch(batch_size)
            train_data = train_data.map(load_batch_tf)
            train_data = train_data.prefetch(1)
            train_loss, train_kacc = process(model, train_data, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

