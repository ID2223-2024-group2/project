import time

import requests
from requests import Response
import typing


def fetch_with_exponential_backoff(url: str, timeout: int, max_retries: int) -> typing.Union[Response, None]:
    retries = 0
    wait_time = timeout

    while retries < max_retries:
        try:
            response = requests.get(url, timeout=timeout)
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectTimeout):
            retries += 1
            print(f"Timeout reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2

    print("Max retries reached. Exiting.")
    return None
