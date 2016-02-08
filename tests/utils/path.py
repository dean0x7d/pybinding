import pathlib


def path_from_fixture(request, prefix, variant='', ext='', override_group=''):
    """Use a fixture's `request` argument to create a unique file path

    The final return path will look like:
        prefix/module_name/test_name[fixture_param]variant.ext

    Parameters
    ----------
    request
        Pytest fixture argument.
    prefix : str
        Path prefix. If a relative path is given it's assumed to be inside the tests dir.
    variant : str, optional
        Appended to the path just before the suffix.
    ext : str, optional
        File name extension
    override_group : str, optional
        'test_name[fixture_param]' -> 'override_group[fixture_param]'

    Returns
    -------
    pathlib.Path
    """
    test_dir = pathlib.Path(str(request.fspath.join('..')))
    module_name = request.module.__name__.split('.')[-1].replace('test_', '')

    name = request.node.name.replace('test_', '') + variant
    if override_group:
        # 'test_name[fixture_param]' -> 'override_name[fixture_param]'
        part = name.partition('[')
        name = override_group + part[1] + part[2]

    return (test_dir / prefix / module_name / name).with_suffix(ext)
