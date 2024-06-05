"""
    This module contains utulity
"""

__author__ = "Tomasz Rybotycki"

import os
from zipfile import ZipFile

from settings import *


def experiments_clean_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        os.remove(job_list_path)
    except Exception as alert:
        print(alert)

