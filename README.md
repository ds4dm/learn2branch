__Update__: check out library [Ecole](https://doc.ecole.ai), which reimplements everything you'll need for learning to branch, in a nice and clean Python package ([paper here](https://arxiv.org/abs/2011.06069)).

# Exact Combinatorial Optimization with Graph Convolutional Neural Networks

Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi

This is the official implementation of our NeurIPS 2019 [paper](https://arxiv.org/abs/1906.01629).

## Installation

See installation instructions [here](INSTALL.md).

## Running the experiments

### Set Covering
```
# Generate MILP instances
python 01_generate_instances.py setcover
# Generate supervised learning datasets
python 02_generate_samples.py setcover -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py setcover -m baseline -s $i
    python 03_train_gcnn.py setcover -m mean_convolution -s $i
    python 03_train_gcnn.py setcover -m no_prenorm -s $i
    python 03_train_competitor.py setcover -m extratrees -s $i
    python 03_train_competitor.py setcover -m svmrank -s $i
    python 03_train_competitor.py setcover -m lambdamart -s $i
done
# Test
python 04_test.py setcover
# Evaluation
python 05_evaluate.py setcover
```

### Combinatorial Auction
```
# Generate MILP instances
python 01_generate_instances.py cauctions
# Generate supervised learning datasets
python 02_generate_samples.py cauctions -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py cauctions -m baseline -s $i
    python 03_train_competitor.py cauctions -m extratrees -s $i
    python 03_train_competitor.py cauctions -m svmrank -s $i
    python 03_train_competitor.py cauctions -m lambdamart -s $i
done
# Test
python 04_test.py cauctions
# Evaluation
python 05_evaluate.py cauctions
```

### Capacitated Facility Location
```
# Generate MILP instances
python 01_generate_instances.py facilities
# Generate supervised learning datasets
python 02_generate_samples.py facilities -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py facilities -m baseline -s $i
    python 03_train_competitor.py facilities -m extratrees -s $i
    python 03_train_competitor.py facilities -m svmrank -s $i
    python 03_train_competitor.py facilities -m lambdamart -s $i
done
# Test
python 04_test.py facilities
# Evaluation
python 05_evaluate.py facilities
```

### Maximum Independent Set
```
# Generate MILP instances
python 01_generate_instances.py indset
# Generate supervised learning datasets
python 02_generate_samples.py indset -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gcnn.py indset -m baseline -s $i
    python 03_train_competitor.py indset -m extratrees -s $i
    python 03_train_competitor.py indset -m svmrank -s $i
    python 03_train_competitor.py indset -m lambdamart -s $i
done
# Test
python 04_test.py indset
# Evaluation
python 05_evaluate.py indset
```

## Citation
Please cite our paper if you use this code in your work.
```
@inproceedings{conf/nips/GasseCFCL19,
  title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
  author={Gasse, Maxime and Chételat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
  booktitle={Advances in Neural Information Processing Systems 32},
  year={2019}
}
```

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.

