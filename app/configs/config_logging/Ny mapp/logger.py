import datetime
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler


class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, log_dir, app_name, when, interval, backupCount, encoding=None, delay=False, utc=False, atTime=None):
        self.log_dir = log_dir
        self.app_name = app_name
        filename = self.get_new_log_file()
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)

    def get_new_log_file(self):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{current_date}_{self.app_name}_log.txt")

    def doRollover(self):
        # override doRollOver to correctly name new log files
        self.stream.close()
        # Get the new log file name
        self.baseFilename = self.get_new_log_file()
        self.stream = self._open()


def config_logger():
    # New handler and Remove the previous handler
    logger = logging.getLogger()

    if logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Load logger configurations
    with open("app/configs/config_logging/logger_config.json", "r") as f:
        config = json.load(f)
    log_dir = config.get("log_dir")
    app_name = config.get("app_name")

    # Create a new file handler for the logger
    file_handler = CustomTimedRotatingFileHandler(log_dir, app_name, when='midnight', interval=1, backupCount=10, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(name)s\t%(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
