"""
    This module is the basis for the witness experiments.
"""

import time

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from qiskit_aer import AerSimulator

from job import Job, VivianiJob
from utils import *


def enqueue_jobs():
    backend_str: str = ""
    generate_json_file(backend_str)

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    qubits_list = [0, 1]  # TR: For tests

    for _ in range(N_JOBS):
        job: Job = VivianiJob()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(qubits_list)
        jobs.append(job)

    save_to_log(LOG_FILE_NAME, "Circuits built")
    save_to_log(LOG_FILE_NAME, "Adding jobs to queue...")

    i: int = 0
    job_list_path: str = f"{RESULTS_FOLDER_NAME}/job_list.csv"

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(service.active_account())
        # backend = service.get_backend('ibm_brisbane')
        # backend = service.get_backend("ibmq_qasm_simulator")
        backend = AerSimulator()

        try:
            sampler = Sampler(backend=backend)
            jobs[i].queued_job = sampler.run(jobs[i].circuits, shots=N_SHOTS)

            # print(jobs[i].queued_job)
            # jobs[i].queued_job = backend.run(jobs[i].circuits, shots=N_SHOTS)

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
            save_to_log(
                LOG_FILE_NAME,
                f"Error adding {str(i)} job to queue. Waiting for the next try...",
            )
            time.sleep(WAIT_TIME)
        ndone = True

        while ndone:
            time.sleep(WAIT_TIME)
            if jobs[i].update_status():
                if jobs[i].last_status == "DONE" and not jobs[i].if_saved:
                    save_to_log(
                        LOG_FILE_NAME,
                        f"{str(i)}. job status: DONE. Saving results to file...",
                    )
                    filename = (
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i + 10 + 11)}.csv"
                    )
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    i += 1
                    ndone = False
                elif jobs[i].last_status in ["ERROR", "CANCELLED"]:
                    save_to_log(
                        LOG_FILE_NAME, f"{str(i)}. job status: {jobs[i].last_status}"
                    )
                    i += 1
                    ndone = False

    experiments_clean_up(job_list_path)

    save_to_log(LOG_FILE_NAME, "Program ended.")


def main():
    print("Start")
    enqueue_jobs()
    print("Done")


if __name__ == "__main__":
    main()
