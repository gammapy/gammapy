# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Useful tutorial to make a Flask apps: https://github.com/hplgit/web4sciapps
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.scripts import get_parser, set_up_logging_from_args

__all__ = []
# __all__ = ['run_data_browser']

# TODO: use Astropy config option to handle this!
DEFAULT_DATADIR = '/Users/deil/work/_Data/hess/HESSFITS/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std/'

# TODO: implement this ... users often just want to browse the observations
# for their target ... the app should have the concept of "current obs selection"
DEFAULT_OBSLIST = 'none'


def main(args=None):
    parser = get_parser()
    parser.add_argument('--data_dir', default=DEFAULT_DATADIR,
                        help='Data directory')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number')
    # parser.add_argument('--obs_list', default=DEFAULT_OBSLIST,
    #                     help='Observation list')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    # We put this import here, because it contains package-level imports
    # of optional dependencies, which would make pytest and sphinx freak
    # out if they found it ...
    from .app import app

    # TODO: communicating via app.config doesn't work!?
    app.config['GAMMAPY_DATA_DIR'] = args.data_dir

    # TODO: args.loglevel doesn't exist!???
    # How to get optional arguments in `args`?
    # app.config['GAMMAPY_LOG_LEVEL'] = args.loglevel


    # Auto-open the app in the default webbrowser after giving
    # it 1 second to start up ...
    # http://stackoverflow.com/a/11126505/498873
    # import webbrowser
    # import threading
    # url = 'http://127.0.0.1:{}/'.format(args.port)
    # threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    # TODO: This doesn't work properly ... a new window is opened
    # by Flask because of the auto-restart feature with `debug=True`.
    # Maybe this solution could work?
    # http://stackoverflow.com/a/2634716/498873

    app.run(debug=True, port=args.port)



