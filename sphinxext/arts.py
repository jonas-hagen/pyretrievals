import urllib.parse
from docutils import nodes, utils


arts_elements = ('group', 'variable', 'method', 'agenda')
arts_path = {el: el+'s' for el in arts_elements}


def make_arts_link(name, rawtext, text, lineno, inliner, options={}, content=[]):
    parts = name.split(':')

    if len(parts) < 2 or parts[1] not in arts_elements:
        msg = inliner.reporter.error(
            'Unknown arts role "{}".'.format(name), line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    kind = parts[1]
    env = inliner.document.settings.env
    docserver_url = env.config.arts_docserver_url.strip('/')
    uri = '/'.join([docserver_url, kind+'s', text])
    node = nodes.reference(rawtext,  utils.unescape(text), refuri=uri, **options)
    return [node], []


def setup(app):
    """Setup function to register the extension"""
    app.add_config_value('arts_docserver_url',
                         'http://radiativetransfer.org/docserver-trunk',
                         'env')
    for kind in arts_elements:
        app.add_role('arts:'+kind, make_arts_link)
