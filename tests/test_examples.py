import pytest
import pathlib


examples_path = (pathlib.Path(__file__) / "../../docs/examples").resolve()
examples = {path.stem: path for path in examples_path.glob('*/*.py')}
assert len(examples) != 0


@pytest.fixture(scope='module', ids=list(examples.keys()), params=examples.values())
def example_file(request):
    return request.param


def test_docs(example_file):
    """Make sure all example files execute without error"""
    filename = str(example_file)
    exec(compile(open(filename, 'rb').read(), filename, 'exec'), {})
