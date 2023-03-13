import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler
from time import strftime, gmtime


class Logger:
    def __init__(self, name):
        self.LOG_PATH = os.getcwd() + '/Apps/ai_intent/log/intent'
        self.log_fmt = '%(asctime)s - File \"%(name)s/%(filename)s\" - line %(lineno)s - %(levelname)s - %(message)s'
        self.formatter = logging.Formatter(self.log_fmt)
        self.log = logging.getLogger(name)

    def getLogger(self):
        self.log.setLevel(logging.INFO)
        self.log.suffix = "%Y-%m-%d_%H-%M.log"
        self.log.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
        self.log_file_handler = TimedRotatingFileHandler(filename=self.LOG_PATH, when="D", interval=1, backupCount=7)
        self.log_file_handler.setFormatter(self.formatter)
        self.log.addHandler(self.log_file_handler)
        return self.log
