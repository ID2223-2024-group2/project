import os
import sys
import warnings

from shared.constants import FeedType, OperatorsWithRT
from shared.api import fetch_with_exponential_backoff

try:
    gtfsr_rt_key = os.environ.get("GTRFSR_RT_API_KEY").strip()
    if not gtfsr_rt_key:
        with open(".gtfsr_rt_key", "r") as f:
            gtfsr_rt_key = f.read().strip()
            if not gtfsr_rt_key:
                raise FileNotFoundError
except FileNotFoundError:
    warnings.warn("No API key found. Please create a .gtfsr_rt_key file with your API key.")
    sys.exit()

try:
    gtfsr_static_key = os.environ.get("GTRFSR_STATIC_API_KEY").strip()
    if not gtfsr_static_key:
        with open(".gtfsr_static_key", "r") as f:
            gtfsr_static_key = f.read().strip()
            if not gtfsr_static_key:
                raise FileNotFoundError
except FileNotFoundError:
    warnings.warn("No API key found. Please create a .gtfsr_static_key file with your API key.")
    sys.exit()

STATIC_URL = "https://opendata.samtrafiken.se/gtfs/{operator}/{operator}.zip?key={api_key}"
REALTIME_URL = "https://opendata.samtrafiken.se/gtfs-rt/{operator}/{feed}.pb?key={api_key}"
DEFAULT_DOWNLOAD_DIR = "./dev_data/gtfsr_download"
# KoDa docs: "creation of an archive can take between 1 and 60 minutes"
API_TIMEOUT = 20
MAX_RETRIES = 5


def get_static_download_path(operator: str, date: str, download_dir=DEFAULT_DOWNLOAD_DIR) -> str:
    return f'{download_dir}/{operator}_static_{date.replace("-", "_")}.7z'


def get_rt_download_path(operator: str, date: str, download_dir=DEFAULT_DOWNLOAD_DIR) -> str:
    return f'{download_dir}/{operator}_rt_{date.replace("-", "_")}.pb'


def fetch_gtfs_archive(url, target_path, force=False):
    if os.path.exists(target_path) and not force:
        print("File already exists.")
        return target_path

    response = fetch_with_exponential_backoff(url, API_TIMEOUT, MAX_RETRIES)
    if response is None:
        return None

    if response.status_code == 200:
        with open(target_path, "wb") as file:
            file.write(response.content)
        print("File is ready.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    return target_path


def fetch_gtfs_static_archive(operator: OperatorsWithRT, date: str, download_dir=DEFAULT_DOWNLOAD_DIR):
    url = STATIC_URL.format(operator=operator.value, date=date, api_key=gtfsr_static_key)
    target_path = get_static_download_path(operator.value, date, download_dir)
    return fetch_gtfs_archive(url, target_path)


def fetch_gtfs_realtime_pb(operator: OperatorsWithRT, feed: FeedType, date: str,
                                download_dir=DEFAULT_DOWNLOAD_DIR, force=False):
    url = REALTIME_URL.format(operator=operator.value, feed=feed.value, date=date, api_key=gtfsr_rt_key)
    target_path = get_rt_download_path(operator.value, date, download_dir)
    return fetch_gtfs_archive(url, target_path, force=force)
