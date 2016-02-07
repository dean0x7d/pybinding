#!/usr/bin/env sh

if [ "$TRAVIS_OS_NAME" = "linux" ] && [ "$PYTHON" = "3.5" ]; then
    # deploy only one sdist: linux with python3.5
    python setup.py sdist
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*.tar.gz
elif [ "$TRAVIS_OS_NAME" = "osx" ]; then
    # deploy wheels for all supported python versions
    python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*.whl
fi
