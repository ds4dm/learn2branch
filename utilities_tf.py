import pickle
import gzip
import numpy as np

import tensorflow as tf


def load_batch_gcnn(sample_files):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []
    candss = []
    cand_choices = []
    cand_scoress = []

    # load samples
    for filename in sample_files:
        with gzip.open(filename, 'rb') as f:
            sample = pickle.load(f)

        sample_state, _, sample_action, sample_cands, cand_scores = sample['data']

        sample_cands = np.array(sample_cands)
        cand_choice = np.where(sample_cands == sample_action)[0][0]  # action index relative to candidates

        c, e, v = sample_state
        c_features.append(c['values'])
        e_indices.append(e['indices'])
        e_features.append(e['values'])
        v_features.append(v['values'])
        candss.append(sample_cands)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cds.shape[0] for cds in candss]

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
    # candidate indices as well
    candss = np.concatenate([cands + shift
        for cands, shift in zip(candss, cv_shift[1])])
    cand_choices = np.array(cand_choices)
    cand_scoress = np.concatenate(cand_scoress, axis=0)

    # convert to tensors
    c_features = tf.convert_to_tensor(c_features, dtype=tf.float32)
    e_indices = tf.convert_to_tensor(e_indices, dtype=tf.int32)
    e_features = tf.convert_to_tensor(e_features, dtype=tf.float32)
    v_features = tf.convert_to_tensor(v_features, dtype=tf.float32)
    n_cs_per_sample = tf.convert_to_tensor(n_cs_per_sample, dtype=tf.int32)
    n_vs_per_sample = tf.convert_to_tensor(n_vs_per_sample, dtype=tf.int32)
    candss = tf.convert_to_tensor(candss, dtype=tf.int32)
    cand_choices = tf.convert_to_tensor(cand_choices, dtype=tf.int32)
    cand_scoress = tf.convert_to_tensor(cand_scoress, dtype=tf.float32)
    n_cands_per_sample = tf.convert_to_tensor(n_cands_per_sample, dtype=tf.int32)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, candss, cand_choices, cand_scoress
