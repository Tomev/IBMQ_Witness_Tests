"""
    This module contains utulity
"""

__author__ = "Tomasz Rybotycki"

import datetime
import json
import os
from zipfile import ZipFile

from src.settings import *


def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        os.remove(job_list_path)
    except Exception as alert:
        print(alert)

