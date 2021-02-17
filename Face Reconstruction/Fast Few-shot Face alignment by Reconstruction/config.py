import os


ENCODER_LAYER_NORMALIZATION = 'batch'
DECODER_LAYER_NORMALIZATION = 'batch'
DECODER_SPECTRAL_NORMALIZATION = False
DECODER_PLANES_PER_BLOCK = 1


__datasets = {}

def register_dataset(cls):
    __datasets[cls.__name__.lower()] = cls


def get_dataset_class(name):
    return __datasets[name]


def get_registered_dataset_names():
    return __datasets


_default_config_files = [
    './local_config.ini',
    '../local_config.ini',
]


def get_dataset_paths(dsname):
    import configargparse
    _parser = configargparse.ArgParser(default_config_files=_default_config_files)
    def add_dataset(dsname):
        _parser.add_argument(f'--{dsname}', default='./', type=str, metavar='PATH',
                            help=f"root directory of dataset {dsname}")
        _parser.add_argument(f'--{dsname}_local', default='', type=str, metavar='PATH',
                            help='path to directory that will be used to store cached data (e.g. crops)')
    add_dataset(dsname)
    _paths = _parser.parse_known_args()[0]
    assert hasattr(_paths, dsname)
    assert hasattr(_paths, dsname+'_local')

    def check_paths(path, path_local):
        """return 'path' if 'path_local' is not defined"""
        if not os.path.exists(path):
            raise IOError(f"Could not find datset {dsname}. Invalid path '{path}'.")
        if not path_local:
            path_local = path
        if not os.path.exists(path_local):
            from csl_common.utils.io_utils import makedirs
            try:
                makedirs(path_local)
            except:
                print(f"Could not create cache directory for dataset {dsname}.")
                raise
        return path, path_local

    return check_paths(_paths.__getattribute__(dsname), _paths.__getattribute__(dsname+'_local'))


def read_local_config():
    import configargparse
    _parser = configargparse.ArgParser(default_config_files=_default_config_files)
    _parser.add_argument(f'--data', default='./data', type=str, metavar='PATH')
    _parser.add_argument(f'--outputs', default='./outputs', type=str, metavar='PATH')
    _paths = _parser.parse_known_args()[0]
    return _paths


_paths = read_local_config()

DATA_DIR = _paths.data
OUTPUT_DIR = _paths.outputs

MODEL_DIR = os.path.join(DATA_DIR, 'models')
SNAPSHOT_DIR = os.path.join(MODEL_DIR, 'snapshots')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
