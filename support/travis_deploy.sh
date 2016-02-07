#!/usr/bin/env sh

if [ "$TRAVIS_OS_NAME" = "linux" ]; then
    python setup.py sdist
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*.tar.gz
fi
