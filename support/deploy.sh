#!/usr/bin/env bash

cd $TRAVIS_BUILD_DIR
pip install -U wheel
pip install twine

if [ "$TRAVIS_OS_NAME" = "linux" ] && [ -d "$TRAVIS_BUILD_DIR/dist" ]; then
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*.tar.gz
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    PB_NATIVE_SIMD=OFF python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*.whl
fi
