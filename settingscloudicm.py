"""
This module contains settings for the project.
"""

from os import environ,remove
from zipfile import ZipFile

# TOKEN and CRN for each collaborator
#ICM
tkicm="XXX"
crnicm="crn:v1:bluemix:public:quantum-computing:us-east:a/d9c641d90d344e5aaa078da8fbf5437f:73d7db05-e693-4acd-9a38-28b9749253b7::"
tcicm=[tkicm,crnicm,"ICM"]







#excl tcjb"
tcset=[tcicm]

N_TEST_CIRCUITS = 0  # Number of test circuits at the beginning of the job.
N_REPETITIONS = 6 # Number of experiments repetitions.
N_JOBS = 2  # Number of jobs we want to submit.
N_SHOTS = 10000
WAIT_TIME = 30  # Delay (in seconds) between checking job status.
SHOULD_RANDOMIZE = True  # Circuits order randomization.

RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-poly-ki"

def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        remove(job_list_path)
    except Exception as alert:
        print(alert)
