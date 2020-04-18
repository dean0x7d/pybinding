"""Adapted from sphinx.ext.autosummary.generate

Modified to only consider module members listed in `__all__` and
only class members listed in `autodoc_allowed_special_members`.

Copyright 2007-2016 by the Sphinx team, https://github.com/sphinx-doc/sphinx/blob/master/AUTHORS
License: BSD, see https://github.com/sphinx-doc/sphinx/blob/master/LICENSE for details.
"""
import os

from jinja2 import FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from sphinx import package_dir
from sphinx.ext.autosummary import import_by_name, get_documenter
from sphinx.jinja2glue import BuiltinTemplateLoader
from sphinx.util.osutil import ensuredir
from sphinx.util.inspect import safe_getattr
from sphinx.ext.autosummary.generate import find_autosummary_in_files
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_members(app, obj, typ, include_public=()):
    __all__ = getattr(obj, '__all__', [])
    skip_all = not __all__
    __all__ += include_public

    items = []
    for name in dir(obj):
        try:
            documenter = get_documenter(app, safe_getattr(obj, name), obj)
        except AttributeError:
            continue
        if documenter.objtype == typ:
            items.append(name)

    public = [x for x in items if x in __all__ or skip_all and not x.startswith('_')]
    # only members with docstrings are considered public
    public = [x for x in public if safe_getattr(obj, x).__doc__]
    return public, items


def generate_autosummary_docs(sources, app, suffix='.rst', output_dir=None,
                              base_path=None, builder=None, template_dir=None):
    showed_sources = list(sorted(sources))
    if len(showed_sources) > 20:
        showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
    logger.info('[autosummary] generating autosummary for: %s' % ', '.join(showed_sources))

    if output_dir:
        logger.info('[autosummary] writing to %s' % output_dir)

    if base_path is not None:
        sources = [os.path.join(base_path, filename) for filename in sources]

    # create our own templating environment
    template_dirs = [os.path.join(package_dir, 'ext', 'autosummary', 'templates')]
    if builder is not None:
        # allow the user to override the templates
        template_loader = BuiltinTemplateLoader()
        template_loader.init(builder, dirs=template_dirs)
    else:
        if template_dir:
            template_dirs.insert(0, template_dir)
        template_loader = FileSystemLoader(template_dirs)
    template_env = SandboxedEnvironment(loader=template_loader)

    # read
    items = find_autosummary_in_files(sources)
    # keep track of new files
    new_files = []
    # write
    for name, path, template_name in sorted(set(items), key=str):
        if path is None:
            continue  # The corresponding autosummary:: directive did not have a :toctree: option

        path = output_dir or os.path.abspath(path)
        ensuredir(path)

        try:
            name, obj, parent, mod_name = import_by_name(name)
        except ImportError as e:
            logger.warning('[autosummary] failed to import %r: %s' % (name, e))
            continue

        fn = os.path.join(path, name + suffix)
        # skip it if it exists
        if os.path.isfile(fn):
            continue

        new_files.append(fn)
        with open(fn, 'w') as f:
            doc = get_documenter(app, obj, parent)

            if template_name is not None:
                template = template_env.get_template(template_name)
            else:
                try:
                    template = template_env.get_template('autosummary/%s.rst' % doc.objtype)
                except TemplateNotFound:
                    template = template_env.get_template('autosummary/base.rst')

            ns = {}
            if doc.objtype == 'module':
                ns['members'] = dir(obj)
                ns['functions'], ns['all_functions'] = get_members(app, obj, 'function')
                ns['classes'], ns['all_classes'] = get_members(app, obj, 'class')
                ns['exceptions'], ns['all_exceptions'] = get_members(app, obj, 'exception')
            elif doc.objtype == 'class':
                ns['members'] = dir(obj)
                include_public = app.config.autodoc_allowed_special_members
                ns['methods'], ns['all_methods'] = get_members(app, obj, 'method', include_public)
                ns['attributes'], ns['all_attributes'] = get_members(app, obj, 'attribute')

            parts = name.split('.')
            if doc.objtype in ('method', 'attribute'):
                mod_name = '.'.join(parts[:-2])
                cls_name = parts[-2]
                obj_name = '.'.join(parts[-2:])
                ns['class'] = cls_name
            else:
                mod_name, obj_name = '.'.join(parts[:-1]), parts[-1]

            ns['fullname'] = name
            ns['module'] = mod_name
            ns['objname'] = obj_name
            ns['name'] = parts[-1]
            ns['objtype'] = doc.objtype
            ns['underline'] = len(name) * '='

            rendered = template.render(**ns)
            f.write(rendered)

    # descend recursively to new files
    if new_files:
        generate_autosummary_docs(new_files, app, suffix=suffix, output_dir=output_dir,
                                  base_path=base_path, builder=builder, template_dir=template_dir)


def process_generate_options(app):
    genfiles = app.config.generate_from_files
    if not genfiles:
        return

    ext = '.rst'
    genfiles = [genfile + (not genfile.endswith(ext) and ext or '') for genfile in genfiles]
    generate_autosummary_docs(genfiles, app, suffix=ext, builder=app.builder, base_path=app.srcdir)


def setup(app):
    app.connect('builder-inited', process_generate_options)
    app.add_config_value('generate_from_files', [], 'env')
    return {'version': "0.1"}
