# SCIP solver

Set-up a desired installation path for SCIP / SoPlex (e.g., `/opt/scip`):
```
export SCIPOPTDIR='/opt/scip'
```

## SoPlex

SoPlex 4.0.1 (free for academic uses)

https://soplex.zib.de/download.php?fname=soplex-4.0.1.tgz

```
tar -xzf soplex-4.0.1.tgz
cd soplex-4.0.1/
mkdir build
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

# SCIP

SCIP 6.0.1 (free for academic uses)

https://scip.zib.de/download.php?fname=scip-6.0.1.tgz

```
tar -xzf scip-6.0.1.tgz
cd scip-6.0.1/
```

Apply patch file in `learn2branch/scip_patch/`

```
patch -p1 < ../learn2branch/scip_patch/vanillafullstrong.patch
```

```
mkdir build
cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
make -C ./build -j 4
make -C ./build install
cd ..
```

For reference, original installation instructions [here](http://scip.zib.de/doc/html/CMAKE.php).

# Python dependencies

Recommended setup: conda + python 3

https://docs.conda.io/en/latest/miniconda.html

## Cython

Required to compile PySCIPOpt and PySVMRank
```
conda install cython
```

## PySCIPOpt

SCIP's python interface (modified version)

```
pip install git+https://github.com/ds4dm/PySCIPOpt.git@ml-branching
```

## ExtraTrees
```
conda install scikit-learn=0.20.2  # ExtraTrees
```

## LambdaMART
```
pip install git+https://github.com/jma127/pyltr@78fa0ebfef67d6594b8415aa5c6136e30a5e3395  # LambdaMART
```

## SVMrank
```
git clone https://github.com/ds4dm/PySVMRank.git
cd PySVMRank
wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz  # get SVMrank original source code
mkdir src/c
tar -xzf svm_rank.tar.gz -C src/c
pip install .
```

## Tensorflow
```
conda install tensorflow-gpu=1.12.0
```
