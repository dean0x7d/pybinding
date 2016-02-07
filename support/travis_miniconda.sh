#!/usr/bin/env sh

VERSION=latest
PREFIX=http://repo.continuum.io/miniconda/Miniconda3
if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    OS=Linux-x86_64
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    OS=MacOSX-x86_64
fi

wget -O miniconda.sh $PREFIX-$VERSION-$OS.sh
chmod +x miniconda.sh

CONDA_DIR=$HOME/miniconda3
./miniconda.sh -b -p $CONDA_DIR
rm miniconda.sh

export PATH=$CONDA_DIR/bin:$PATH
conda update --yes conda
conda install --yes python=$PYTHON nomkl numpy scipy matplotlib cmake

# needed for deployment
pip install -U wheel  # probably upgrades conda version
pip install twine
