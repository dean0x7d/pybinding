import pytest
import pathlib


docs = pathlib.Path(__file__).parent
examples = list(docs.glob("tutorial/**/*.py")) + list(docs.glob("examples/**/*.py"))
assert len(examples) != 0


@pytest.fixture(scope='module', ids=[e.stem for e in examples], params=examples)
def example_file(request):
    """An example file from the documentation directory"""
    return request.param


def test_docs(example_file):
    """Make sure all example files execute without error"""
    filename = str(example_file)
    with open(filename, "rb") as file:
        exec(compile(file.read(), filename, "exec"), {})
