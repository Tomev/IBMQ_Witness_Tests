"""
    Module description...
"""

import sys
import time

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService

from src.job import VivianiJob
from src.utils import *

MODULE_FULL_PATH = "/home/jovyan/"
sys.path.insert(1, MODULE_FULL_PATH)


def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    qubits_list = [0, 1]  # TR: For tests

    for _ in range(N_JOBS):
        job = VivianiJob()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(qubits_list)
        jobs.append(job)

    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list.csv"

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(service.active_account())
        # backend = service.get_backend('ibm_brisbane')
        backend = service.get_backend("ibmq_qasm_simulator")
        # backend = GenericBackendV2(num_qubits=2)  # TR: For tests

        try:
            jobs[i].queued_job = backend.run(jobs[i].circuits, shots=N_SHOTS)

            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                "pars": jobs[i].indices_list,
                "token_id": TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)].split("_")[-1]
            }
            job_list_table = pd.concat(
                [job_list_table, pd.DataFrame([job_data])], ignore_index=True
            )
            job_list_table.to_csv(job_list_path)
        except Exception as alert:
            print(alert)
            time.sleep(WAIT_TIME)
        ndone = True

        while ndone:
            time.sleep(WAIT_TIME)
            if jobs[i].update_status():
                if jobs[i].last_status == "DONE" and not jobs[i].if_saved:
                    filename = (
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i)}.csv"
                    )
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    i += 1
                    ndone = False
                elif jobs[i].last_status in ["ERROR", "CANCELLED"]:
                    i += 1
                    ndone = False

    experiments_cleen_up(job_list_path)

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
