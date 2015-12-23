import pytest
import pathlib


docs = (pathlib.Path(__file__) / "../../docs/").resolve()
examples = list(docs.glob('*/**/*.py'))
assert len(examples) != 0


@pytest.fixture(scope='module', ids=[e.stem for e in examples], params=examples)
def example_file(request):
    return request.param


def test_docs(example_file):
    """Make sure all example files execute without error"""
    filename = str(example_file)
    exec(compile(open(filename, 'rb').read(), filename, 'exec'), {})
