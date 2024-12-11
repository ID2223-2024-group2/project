import os

import requests

try:
    with open(".koda_key", "r") as f:
        koda_api_key = f.read()
except FileNotFoundError:
    print("No API key found. Please create a .koda_key file with your API key.")
    exit()

STATIC_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-static/{operator}?date={date}&key={api_key}"
REALTIME_URL = "https://api.koda.trafiklab.se/KoDa/api/v2/gtfs-rt/{operator}/{feed}?date={date}&key={api_key}"
DOWNLOAD_DIR = "./dev_data/koda_download"
MAX_POLL_TIMES = 20

def fetch_gtfs_data(url, target_path):
    if os.path.exists(target_path): # TODO: Check if unzipped files exist too
        print("File already exists.")
        return target_path

    response = requests.get(url)
    if response.status_code == 200:
        with open(target_path, "wb") as file:
            file.write(response.content)
        print("File is ready.")
    elif response.status_code == 202:
        print("File is not ready yet.")
        polled = 0
        while response.status_code == 202 and polled < MAX_POLL_TIMES:
            response = requests.get(url)
            polled += 1
            print(f"Polling {polled}/{MAX_POLL_TIMES}")
        with open(target_path, "wb") as file:
            file.write(response.content)
        print("File is ready.")
    else:
        print(f"Error: {response.status_code}")
    return target_path

def fetch_gtfs_static(operator, date, download_dir = DOWNLOAD_DIR):
    url = STATIC_URL.format(operator=operator, date=date, api_key=koda_api_key)
    target_file = f"{operator}-gtfs-static-{date}.7z"
    target_path = f"{download_dir}/{target_file}"
    return fetch_gtfs_data(url, target_path)

def fetch_gtfs_realtime(operator, feed, date, download_dir = DOWNLOAD_DIR):
    url = REALTIME_URL.format(operator=operator, feed=feed, date=date, api_key=koda_api_key)
    target_file = f"{operator}-gtfs-realtime-{date}.7z"
    target_path = f"{download_dir}/{target_file}"
    return fetch_gtfs_data(url, target_path)