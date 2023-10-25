import datetime
import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler


def config_logger():

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # New handler and Remove the previous handler
        logger = logging.getLogger()
        if logger.handlers:
            logger.removeHandler(logger.handlers[0])

        # Load logger configurations
        with open("app/configs/config_logging/logger_config.json", "r") as f:
            config = json.load(f)

        log_dir = config.get("log_dir")
        app_name = config.get("app_name")

        # log file in directory
        log_file_path = os.path.join(log_dir, f"{current_date}_{app_name}_log.txt")

        # Create a new file handler for the logger
        # file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler = TimedRotatingFileHandler(log_file_path, when='midnight', interval=1, backupCount=10, encoding="utf-8")

        formatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(name)s\t%(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

