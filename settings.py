"""
This module contains settings for the project.
"""

from os import environ,remove
from zipfile import ZipFile

# IBMQ token environmental variables
# naming convention: IBMQ_Token_<TOKEN_ID>
# "IBMQ_Token_KM","IBMQ_Token_MS",
TOKEN_VARIABLES = ["IBMQ_Token_AB"]
TOKENS = {key: environ[key] for key in TOKEN_VARIABLES}

N_TEST_CIRCUITS = 0  # Number of test circuits at the beginning of the job.
N_REPETITIONS = 75  # Number of experiments repetitions.
N_JOBS = 5  # Number of jobs we want to submit.
N_SHOTS = 20000
WAIT_TIME = 30  # Delay (in seconds) between checking job status.
SHOULD_RANDOMIZE = True  # Circuits order randomization.
AUTHOR_INITIALS = "TB"
DESCRIPTION = "Short description of a job"

RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-bell55-multi"

def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        remove(job_list_path)
    except Exception as alert:
        print(alert)
