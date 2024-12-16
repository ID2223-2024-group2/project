import logging


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
