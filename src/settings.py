"""
This module contains settings for the project.
"""

from os import environ

# IBMQ token environmental variables
# naming convention: IBMQ_Token_<TOKEN_ID>
TOKEN_VARIABLES = ["IBMQ_Token_TR", "IBMQ_Token_TR-CFT"]
TOKENS = {key: environ[key] for key in TOKEN_VARIABLES}

N_TEST_CIRCUITS = 0  # Number of test circuits at the beginning of the job.
N_REPETITIONS = 10  # Number of experiments repetitions.
N_JOBS = 2  # Number of jobs we want to submit.
N_SHOTS = 10000
WAIT_TIME = 60  # Delay (in seconds) between checking job status.
SHOULD_RANDOMIZE = True  # Circuits order randomization.
AUTHOR_INITIALS = "TB"
DESCRIPTION = "Short description of a job"

LOG_FILE_NAME = "log.txt"
RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-wit-multi"
