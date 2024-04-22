"""
    Module description...
"""

import sys
import datetime
import json
import time
import os
import pandas as pd
from zipfile import ZipFile
from Jobmulti import save_to_log, WitnessJob
from settings import *
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.fake_provider import GenericBackendV2

MODULE_FULL_PATH = '/home/jovyan/'
sys.path.insert(1, MODULE_FULL_PATH)

backend = None

# ------------------------------------- Funkcje --------------------------------------
# Generowanie pliku json z danymi i zapakowanie go do zipa
def generate_json_file():
    json_data = {
        "author": AUTHOR_INITIALS,
        "jobs": N_JOBS,
        "tests circuits": N_TEST_CIRCUITS,
        "repetitions": N_REPETITIONS,
        "shots": N_SHOTS,
        "backend": str(backend),
        "randomization": SHOULD_RANDOMIZE,
        "script version": "skrypt_v3_TR",
        "start time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": DESCRIPTION
    }

    # Ensure that the results folder exists
    if not os.path.exists(RESULTS_FOLDER_NAME):
        os.makedirs(RESULTS_FOLDER_NAME)

    json_file_path = RESULTS_FOLDER_NAME + "/data.json"

    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file)

    with ZipFile(ZIP_FILE_NAME + '.zip', 'a') as file_zip:
        file_zip.write(json_file_path, arcname='results/data.json')
    try:
        os.remove(json_file_path)
    except:
        save_to_log(LOG_FILE_NAME, "Error removing json file")


def run_scripts():
    generate_json_file()

    jobs = []
    job_list_table = pd.DataFrame()

    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    qubits_list = [0, 1]  # TR: For tests

    for _ in range(N_JOBS):
        job = WitnessJob()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(qubits_list)
        jobs.append(job)
    # -------------------------------- Budowanie układów -------------------------------

    save_to_log(LOG_FILE_NAME, "Circuits built")
    save_to_log(LOG_FILE_NAME, "Adding jobs to queue...")

    # ---------------------------- Ustawianie jobow w kolejce --------------------------
    i = 0
    job_list_path = f'{RESULTS_FOLDER_NAME}/job_list.csv'

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(service.active_account())
        # backend = service.get_backend('ibm_brisbane')
        backend = service.get_backend('ibmq_qasm_simulator')
        backend = GenericBackendV2(num_qubits=2)  # TR: For tests

        try:
            jobs[i].queued_job = backend.run(jobs[i].circuits,
                                                      shots=N_SHOTS)

            # Dodawanie danych joba do tabeli
            job_data = {'job_id': jobs[i].queued_job.job_id(),
                        'pars': jobs[i].indices_list}
            job_list_table = pd.concat([job_list_table, pd.DataFrame([job_data])],
                                       ignore_index=True)
            # job_list_table = job_list_table.append(job_data, ignore_index=True)
            job_list_table.to_csv(job_list_path)
        except Exception as alert:
            print(alert)
            save_to_log(LOG_FILE_NAME, "Error adding " + str(i)
                        + ". job to queue. Waiting for the next try...")
            time.sleep(WAIT_TIME)
        ndone = True
        while ndone:
            time.sleep(WAIT_TIME)
            if jobs[i].update_status():
                if jobs[i].last_status == 'DONE' and not jobs[i].if_saved:
                    save_to_log(LOG_FILE_NAME,
                                f'{str(i)}. job status: DONE. Saving results to file...')
                    filename = f'{RESULTS_FOLDER_NAME}/results_tests_{str(i + 10 + 11)}.csv'
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    i += 1
                    ndone = False
                elif jobs[i].last_status in ['ERROR', 'CANCELLED']:
                    save_to_log(LOG_FILE_NAME,
                                f'{str(i)}. job status: {jobs[i].last_status}')
                    i += 1
                    ndone = False

    # Usuwanie pliku z listą jobów
    with ZipFile(ZIP_FILE_NAME + '.zip', 'a') as zip_file:
        zip_file.write(job_list_path, arcname='results/job_list.csv')

    try:
        os.remove(job_list_path)
    except:
        save_to_log(LOG_FILE_NAME, f'Error removing {job_list_path}')

    save_to_log(LOG_FILE_NAME, "Program ended.")


def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
