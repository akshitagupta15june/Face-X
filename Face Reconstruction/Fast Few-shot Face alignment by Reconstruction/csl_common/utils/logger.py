import sys
import logging
from logging import INFO, ERROR, DEBUG, CRITICAL, FATAL, WARNING, WARN, NOTSET

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': RED,
    'ERROR': RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg, datefmt='%Y-%m-%d %H:%M:%S')
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30+COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class Logger(logging.Logger):
    def __init__(self, name, project=None, level=logging.NOTSET, userId=None):
        logging.Logger.__init__(self, name, level)

        # set up writing to file
        import os
        from os import makedirs as mkdirs
        import datetime
        from os.path import expanduser
        import platform
        # hostname = os.uname()[1]
        hostname = platform.uname().node
        log_dir = os.path.join(expanduser("~"), '.devlogs')
        mkdirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, '{}_{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), hostname))
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        self.addHandler(fh)

        # settings for command line print outs
        __ch = logging.StreamHandler(sys.stdout)
        # __ch.setLevel(logging.DEBUG)
        # FORMAT = RESET_SEQ+"[%(asctime)s][%(levelname)s] %(message)s"+RESET_SEQ
        FORMAT = RESET_SEQ+"[%(asctime)s] %(message)s"+RESET_SEQ
        __formatter = ColoredFormatter(FORMAT)
        __ch.setFormatter(__formatter)
        # __ch.setFormatter(logging.Formatter(FORMAT))
        self.addHandler(__ch)

    def setContext(self, key, value):
        self.context[key] = value

    def log(self, msg,  level, *args, **kwargs):
        extra = kwargs
        extra.update(self.context)
        # super(Logger, self).info(msg)

    def error(self, msg, *args, **kwargs):
        super(Logger, self).error(msg, *args, **kwargs)

    def exception(self, *args, **kwargs):
        e = sys.exc_info()
        extra = kwargs
        extra.update(self.context)
        super(Logger, self).error(e[1].message)
