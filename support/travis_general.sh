#!/usr/bin/env sh

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    export CXX=g++-4.8 CC=gcc-4.8
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    export CXX=clang++ CC=clang
fi

export PB_WERROR=1  # make warnings into errors
export MAKEFLAGS=-j2
