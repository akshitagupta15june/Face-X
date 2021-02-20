from csl_common.utils.logger import *

__vislogger = Logger(__name__)

def debug(msg, *args, **kwargs):
    __vislogger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    __vislogger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    __vislogger.warning(msg, *args, **kwargs)

def log(msg, level='info', *args, **kwargs):
    __vislogger.log(msg, level, *args, **kwargs)

def error(msg, *args, **kwargs):
    __vislogger.error(msg, *args, **kwargs)

def exception(*args, **kwargs):
    __vislogger.exception(*args, **kwargs)

def setLevel(level):
    __vislogger.setLevel(level)

def setLogFile(filepath):
    __vislogger.setLogFile(filepath)

