import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler
from time import strftime, gmtime


class Logger:
    def __init__(self, name):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - File \"%(name)s/%(filename)s\" - line %(lineno)s - %(levelname)s - %(message)s')
        self.LOG_PATH = os.getcwd() + '/Apps/ai_matter_content/log/matter_content'
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


def clean_log():
    path = os.getcwd() + '/Apps/ai_matter_content/log/'
    for i in os.listdir(path):
        if len(i) < 16:
            continue
        file_path = path + i  # 生成日志文件的路径
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        # 获取日志的年月，和今天的年月
        today_m = int(timestamp[4:6])  # 今天的月份
        file_m = int(i[12:14])  # 日志的月份
        today_y = int(timestamp[0:4])  # 今天的年份
        file_y = int(i[7:11])  # 日志的年份
        # 对上个月的日志进行清理，即删除。
        # print(file_path)
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)
