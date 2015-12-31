#!/usr/bin/env sh

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    wget --no-check-certificate -O cmake-install.sh https://cmake.org/files/v3.4/cmake-3.4.1-Linux-x86_64.sh
    chmod +x cmake-install.sh
    
    CMAKE_DIR=$HOME/cmake
    mkdir $CMAKE_DIR
    ./cmake-install.sh --skip-license --prefix=$CMAKE_DIR
    rm cmake-install.sh
    
    export PATH=$CMAKE_DIR/bin:$PATH
fi

cmake --version
