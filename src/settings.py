"""
This module contains settings for the project.
"""
from os import environ

# IBMQ token variables
TOKEN_VARIABLES = ["IBMQ_Token_TR", "IBMQ_Token_TR-CFT"]
TOKENS = {key: environ[key] for key in TOKEN_VARIABLES}

N_TEST_CIRCUITS = 0  # ile układów testujących błąd pomiaru, na początku joba
N_REPETITIONS = 10 # max_experiments nie działa dla symulatora
N_JOBS = 1  # ile jobów wykonujemy
N_SHOTS = 10000
WAIT_TIME = 60  # czas, w sekundach, oczekiwania na ponowną próbę dodania joba do kolejki po niepowodzeniu
SHOULD_RANDOMIZE = True  # randomizacja kątów włączona/wyłączona
AUTHOR_INITIALS = "TB"
DESCRIPTION = "Short description of a job"

LOG_FILE_NAME = "log.txt"
RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-wit-multi"
