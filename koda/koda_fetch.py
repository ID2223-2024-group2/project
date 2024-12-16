import os
import sys
import time
import warnings

import requests

from koda.koda_constants import OperatorsWithRT, FeedType
from shared.api import fetch_with_exponential_backoff

try:
    koda_api_key = os.environ.get("KODA_KEY")
    if not koda_api_key:
        with open(".koda_key", "r") as f:
            koda_api_key = f.read()
            if not koda_api_key:
                raise FileNotFoundError
            # print("API key found in .koda_key file.")
except FileNotFoundError:
    warnings.warn("No API key found. Please create a .koda_key file with your API key.")
    sys.exit()

STATIC_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/{operator}?date={date}&key={api_key}"
REALTIME_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/{operator}/{feed}?date={date}&key={api_key}"
DEFAULT_DOWNLOAD_DIR = "./dev_data/koda_download"
# KoDa docs: "creation of an archive can take between 1 and 60 minutes"
MAX_POLL_TIMES = 120
POLL_DELAY = 30
KODA_API_TIMEOUT = 20
KODA_MAX_RETRIES = 5


def get_static_download_path(operator: str, date: str, download_dir=DEFAULT_DOWNLOAD_DIR) -> str:
    return f'{download_dir}/{operator}_static_{date.replace("-", "_")}.7z'


def get_rt_download_path(operator: str, date: str, download_dir=DEFAULT_DOWNLOAD_DIR) -> str:
    return f'{download_dir}/{operator}_rt_{date.replace("-", "_")}.7z'


def fetch_gtfs_archive(url, target_path):
    if os.path.exists(target_path):
        print("File already exists.")
        return target_path

    response = fetch_with_exponential_backoff(url, KODA_API_TIMEOUT, KODA_MAX_RETRIES)
    if response.status_code == 200:
        with open(target_path, "wb") as file:
            file.write(response.content)
        print("File is ready.")
    elif response.status_code == 202:
        print("File is not ready yet.")
        polled = 0
        while response.status_code == 202 and polled < MAX_POLL_TIMES:
            response = requests.get(url, timeout=KODA_API_TIMEOUT)
            polled += 1
            print(f"Polling {polled}/{MAX_POLL_TIMES}")
            time.sleep(POLL_DELAY)
        if response.status_code == 202:
            print("Timeout reached.")
            return None
        with open(target_path, "wb") as file:
            file.write(response.content)
        print("File is ready.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    return target_path


def fetch_gtfs_static_archive(operator: OperatorsWithRT, date: str, download_dir=DEFAULT_DOWNLOAD_DIR):
    url = STATIC_URL.format(operator=operator.value, date=date, api_key=koda_api_key)
    target_path = get_static_download_path(operator.value, date, download_dir)
    return fetch_gtfs_archive(url, target_path)


def fetch_gtfs_realtime_archive(operator: OperatorsWithRT, feed: FeedType, date: str,
                                download_dir=DEFAULT_DOWNLOAD_DIR):
    url = REALTIME_URL.format(operator=operator.value, feed=feed.value, date=date, api_key=koda_api_key)
    target_path = get_rt_download_path(operator.value, date, download_dir)
    return fetch_gtfs_archive(url, target_path)
