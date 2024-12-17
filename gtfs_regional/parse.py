DATA_DIR = "./dev_data/gtfsr_data"


def get_static_dir_path(operator: str, date: str, data_dir=DATA_DIR) -> str:
    return f'{data_dir}/{operator}_static_{date.replace("-", "_")}'