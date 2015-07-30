import pytest

import pathlib
import pybinding as pb

mock_data = list(range(10))


def round_trip(obj, file):
    pb.save(obj, file)
    return pb.load(file) == obj


def test_path_type(tmpdir):
    # str
    file = str(tmpdir / 'file')
    assert round_trip(mock_data, file)

    # py.path.local object
    file = tmpdir / 'file'
    assert round_trip(mock_data, file)

    # pathlib object
    file = pathlib.Path(str(tmpdir)) / 'file'
    assert round_trip(mock_data, file)

    # file object
    file = str(tmpdir / 'file')
    with open(file, 'wb') as f:
        pb.save(mock_data, f)
    with open(file, 'rb') as f:
        assert mock_data == pb.load(f)


def test_extension(tmpdir):
    file = tmpdir / 'file'
    pb.save(mock_data, file)
    assert mock_data == pb.load(tmpdir / 'file.pbz')

    file = tmpdir / 'file.ext'
    pb.save(mock_data, file)
    assert mock_data == pb.load(tmpdir / 'file.ext')
