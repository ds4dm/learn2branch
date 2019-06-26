# Exact Combinatorial Optimization with Graph Convolutional Neural Networks

Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi

https://arxiv.org/abs/1906.01629

## Installing the dependencies

Recommended: conda + python 3

https://docs.conda.io/en/latest/miniconda.html

### Tensorflow
```
conda install tensorflow-gpu=1.12.0
```

### SCIP

SoPlex 4.0.1 (free for academic uses)

https://soplex.zib.de/download.php?fname=soplex-4.0.1.tgz

```
tar -xzf soplex-4.0.1.tgz
cd soplex-4.0.1/
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="YOUR_INSTALL_DIR"
make -C ./build -j 4
make -C ./build install
cd ..
```

SCIP 6.0.1 (free for academic uses)

https://scip.zib.de/download.php?fname=scip-6.0.1.tgz

```
tar -xzf scip-6.0.1.tgz
cd scip-6.0.1/
```

Apply patches in `learn2branch/scip_patch/`

```
patch -p1 ../learn2branch/scip_patch/0001-vanillafullstrong-branching-rule-initial-version.patch
patch -p1 ../learn2branch/scip_patch/0002-Vanillafullstrong-bugfixes.patch
patch -p1 ../learn2branch/scip_patch/0003-vanillafulltrong-implemented-indempotent-functionali.patch
patch -p1 ../learn2branch/scip_patch/0004-Compilation-fix.patch
```

```
mkdir build
cmake -S . -B build -DSOPLEX_DIR="YOUR_INSTALL_DIR" -DCMAKE_INSTALL_PREFIX="YOUR_INSTALL_DIR"
make -C ./build -j 4
make -C ./build install
cd ..
```

Original installation instructions here:

http://scip.zib.de/doc/html/CMAKE.php

### Cython

Required for PySCIPOpt and PySVMRank
```
conda install cython
```

### PySCIPOpt

SCIP's python interface (modified version)

```
SCIPOPTDIR="YOUR_INSTALL_DIR" pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching
```

### ExtraTrees
```
conda install scikit-learn=0.20.2  # ExtraTrees
```

### LambdaMART
```
pip install git+https://github.com/jma127/pyltr@78fa0ebfef67d6594b8415aa5c6136e30a5e3395  # LambdaMART
```

### SVMrank
```
git clone https://github.com/ds4dm/PySVMRank.git
cd PySVMRank
wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz  # get SVMrank original source code
mkdir src/c
tar -xzf svm_rank.tar.gz -C src/c
pip install .
```

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

