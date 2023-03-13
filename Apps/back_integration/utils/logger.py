import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler
from time import strftime, gmtime


class Logger:
    def __init__(self, name):
        self.log_path = os.getcwd() + '/Apps/back_integration/log/diagnose'
        self.log_fmt = '%(asctime)s - File \"%(name)s/%(filename)s\" - line %(lineno)s - %(levelname)s - %(message)s'
        self.formatter = logging.Formatter(self.log_fmt)
        self.logger = logging.getLogger(name)

    def getLogger(self):
        self.logger.setLevel(logging.INFO)
        self.logger.suffix = "%Y-%m-%d_%H-%M.log"
        self.logger.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
        log_file_handler = TimedRotatingFileHandler(filename=self.log_path, when="D", interval=1, backupCount=7)
        log_file_handler.setFormatter(self.formatter)
        self.logger.addHandler(log_file_handler)
        return self.logger


def clean_log():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    for i in os.listdir(path):
        if not i.startswith("goal_set_"):
            continue
        file_path = os.path.join(path, i)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        today_m = int(timestamp[4:6])  # 今天的月份
        today_d = int(timestamp[6:8])  # 今天的日期
        t = os.path.getmtime(file_path)
        timeStruce = time.localtime(t)
        times = time.strftime('%Y-%m-%d%H:%M:%S', timeStruce)
        file_m = int(times[5:7])  # 日志的月份
        file_d = int(times[8:10])  # 日志的日期
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_d < today_d - 5:
            if os.path.exists(file_path):
                os.remove(file_path)
