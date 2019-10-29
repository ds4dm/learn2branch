import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle

import pyscipopt as scip

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import svmrank

import utilities


class PolicyBranching(scip.Branchrule):

    def __init__(self, policy):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']

        if self.policy_type == 'gcnn':
            model = policy['model']
            model.restore_state(policy['parameters'])
            self.policy = tfe.defun(model.call, input_signature=model.input_signature)

        elif self.policy_type == 'internal':
            self.policy = policy['name']

        elif self.policy_type == 'ml-competitor':
            self.policy = policy['model']

            # feature parameterization
            self.feat_shift = policy['feat_shift']
            self.feat_scale = policy['feat_scale']
            self.feat_specs = policy['feat_specs']

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)

        # custom policy branching
        else:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            # initialize root buffer for Khalil features extraction
            if self.model.getNNodes() == 1 \
                    and self.policy_type == 'ml-competitor' \
                    and self.feat_specs['type'] in ('khalil', 'all'):
                utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]

            elif self.policy_type == 'gcnn':
                state = utilities.extract_state(self.model, self.state_buffer)

                # convert state to tensors
                c, e, v = state
                state = (
                    tf.convert_to_tensor(c['values'], dtype=tf.float32),
                    tf.convert_to_tensor(e['indices'], dtype=tf.int32),
                    tf.convert_to_tensor(e['values'], dtype=tf.float32),
                    tf.convert_to_tensor(v['values'], dtype=tf.float32),
                    tf.convert_to_tensor([c['values'].shape[0]], dtype=tf.int32),
                    tf.convert_to_tensor([v['values'].shape[0]], dtype=tf.int32),
                )

                var_logits = self.policy(state, tf.convert_to_tensor(False)).numpy().squeeze(0)

                candidate_scores = var_logits[candidate_mask]
                best_var = candidate_vars[candidate_scores.argmax()]

            elif self.policy_type == 'ml-competitor':

                # build candidate features
                candidate_states = []
                if self.feat_specs['type'] in ('all', 'gcnn_agg'):
                    state = utilities.extract_state(self.model, self.state_buffer)
                    candidate_states.append(utilities.compute_extended_variable_features(state, candidate_mask))
                if self.feat_specs['type'] in ('all', 'khalil'):
                    candidate_states.append(utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer))
                candidate_states = np.concatenate(candidate_states, axis=1)

                # feature preprocessing
                candidate_states = utilities.preprocess_variable_features(candidate_states, self.feat_specs['augment'], self.feat_specs['qbnorm'])

                # feature normalization
                candidate_states =  (candidate_states - self.feat_shift) / self.feat_scale

                candidate_scores = self.policy.predict(candidate_states)
                best_var = candidate_vars[candidate_scores.argmax()]

            else:
                raise NotImplementedError

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    result_file = f"{args.problem}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    instances = []
    seeds = [0, 1, 2, 3, 4]
    gcnn_models = ['baseline']
    other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    internal_branchers = ['relpscost']
    time_limit = 3600

    if args.problem == 'setcover':
        instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
        gcnn_models += ['mean_convolution', 'no_prenorm']

    elif args.problem == 'cauctions':
        instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'facilities':
        instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(20)]

    elif args.problem == 'indset':
        instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_500_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(20)]
        instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(20)]

    else:
        raise NotImplementedError

    branching_policies = []

    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })
    # ML baselines
    for model in other_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'ml-competitor',
                'name': model,
                'seed': seed,
                'model': f'trained_models/{args.problem}/{model}/{seed}',
            })
    # GCNN models
    for model in gcnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gcnn',
                'name': model,
                'seed': seed,
                'parameters': f'trained_models/{args.problem}/{model}/{seed}/best_params.pkl'
            })

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                import model
                importlib.reload(model)
                loaded_models[policy['name']] = model.GCNPolicy()
                del sys.path[0]
            policy['model'] = loaded_models[policy['name']]

    # load ml-competitor models
    for policy in branching_policies:
        if policy['type'] == 'ml-competitor':
            try:
                with open(f"{policy['model']}/normalization.pkl", 'rb') as f:
                    policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
            except:
                policy['feat_shift'], policy['feat_scale'] = 0, 1

            with open(f"{policy['model']}/feat_specs.pkl", 'rb') as f:
                policy['feat_specs'] = pickle.load(f)

            if policy['name'].startswith('svmrank'):
                policy['model'] = svmrank.Model().read(f"{policy['model']}/model.txt")
            else:
                with open(f"{policy['model']}/model.pkl", 'rb') as f:
                    policy['model'] = pickle.load(f)

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
    ]
    os.makedirs('results', exist_ok=True)
    with open(f"results/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                tf.set_random_seed(policy['seed'])

                m = scip.Model()
                m.setIntParam('display/verblevel', 0)
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', time_limit)

                brancher = PolicyBranching(policy)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                })

                csvfile.flush()
                m.freeProb()

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")

