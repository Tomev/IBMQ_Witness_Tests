"""
    This module contains utulity
"""

__author__ = "Tomasz Rybotycki"

import datetime
import json
import os
from zipfile import ZipFile

from src.settings import *


def save_to_log(filename: str, text: str) -> None:
    data = datetime.datetime.now()
    message = f"{data.strftime('%Y-%m-%d %H:%M:%S')}: {text}"

    log = open(filename, "a")
    log.write(message + "\n")
    log.close()

    print(message)


def generate_json_file(backend: str):
    json_data = {
        "author": AUTHOR_INITIALS,
        "jobs": N_JOBS,
        "tests circuits": N_TEST_CIRCUITS,
        "repetitions": N_REPETITIONS,
        "shots": N_SHOTS,
        "backend": backend,
        "randomization": SHOULD_RANDOMIZE,
        "script version": "skrypt_v3_TR",
        "start time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": DESCRIPTION,
    }

    # Ensure that the results folder exists
    if not os.path.exists(RESULTS_FOLDER_NAME):
        os.makedirs(RESULTS_FOLDER_NAME)

    json_file_path = RESULTS_FOLDER_NAME + "/data.json"

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file)

    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as file_zip:
        file_zip.write(json_file_path, arcname="results/data.json")
    try:
        os.remove(json_file_path)
    except:
        save_to_log(LOG_FILE_NAME, "Error removing json file")


def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        os.remove(job_list_path)
    except FileExistsError:
        save_to_log(LOG_FILE_NAME, f"Error removing {job_list_path}")
