import os

DATA_DIR = "./dev_data/koda_data"


def get_rt_dir_path(operator: str, date: str, data_dir=DATA_DIR) -> str:
    return f'{data_dir}/{operator}_rt_{date.replace("-", "_")}'


def get_rt_dir_info(rt_dir_path: str):
    parts = rt_dir_path.split("/")
    operator = parts[-1].split("_")[0]
    date = "_".join(parts[-1].split("_")[2:])
    return operator, date


def get_rt_hour_dir_path(operator: str, feed_type: str, date: str, hour: int, data_dir=DATA_DIR) -> str:
    year, month, day = date.split("-")
    hour_filled = str(hour).zfill(2)
    rt_dir_path = get_rt_dir_path(operator, date, data_dir)
    subfolder_path = os.path.join(operator, feed_type, year, month, day, hour_filled)
    hour_dir_path = os.path.join(rt_dir_path, subfolder_path)
    return hour_dir_path


def get_pb_file_path(operator: str, feed_type: str, date: str, hour: int, minute: int, second: int, data_dir=DATA_DIR):
    year, month, day = date.split("-")
    hour_filled = str(hour).zfill(2)
    minute_filled = str(minute).zfill(2)
    second_filled = str(second).zfill(2)
    file_name = f"{operator}-{feed_type.lower()}-{year}-{month}-{day}T{hour_filled}-{minute_filled}-{second_filled}Z.pb"
    hour_dir_path = get_rt_hour_dir_path(operator, feed_type, date, hour, data_dir)
    file_path = os.path.join(hour_dir_path, file_name)
    return file_path


def get_pb_file_info(file_path: str):
    parts = file_path.split("/")
    operator = parts[-7]
    feed_type = parts[-6]
    date = "-".join(parts[-5:-2])
    hour, minute, second = parts[-2].split("-")
    return operator, feed_type, date, int(hour), int(minute), int(second)