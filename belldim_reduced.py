"""
    Module description...
"""

import sys
import time
import matplotlib.pyplot as plt

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2,FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.compiler import transpile, schedule
import qiskit.pulse as pulse
from jobsampler_reduced import BellJob
from os import environ,remove

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler
from zipfile import ZipFile
RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-bell-aqt"
N_SHOTS = 200
N_REPETITIONS = 2
WAIT_TIME = 30
SHOULD_RANDOMIZE = True
N_JOBS = 1
def title_generator(program, device):
    return ''
def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        remove(job_list_path)
    except Exception as alert:
        print(alert)

def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    qubit=4

    for _ in range(N_JOBS):
        job = BellJob()
        job.n_repetitions = N_REPETITIONS
        job.add_witness_circuits(qubit)
        jobs.append(job)

    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list_aqt.csv"
    provider = AQTProvider("ACCESS_TOKEN")
    backend = provider.get_backend("offline_simulator_no_noise")
    sampler = AQTSampler(backend=backend)
    sampler.set_transpile_options(optimization_level=0)

    while i < N_JOBS:
        try:
            jobs[i].queued_job = sampler.run(jobs[i].circuits, shots=N_SHOTS)

            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                #"token_id": TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)].split("_")[-1]
            }
            job_list_table = pd.concat(
                [job_list_table, pd.DataFrame([job_data])], ignore_index=True
            )
            job_list_table.to_csv(job_list_path)
            i += 1
        except Exception as alert:
            print(alert)
            time.sleep(WAIT_TIME)
    ndone = True

    while ndone:
        ndone = False
        time.sleep(WAIT_TIME)
        for i in range(N_JOBS):
            if jobs[i].update_status():
                print(i,jobs[i].last_status)
                if jobs[i].last_status == "DONE" and not jobs[i].if_saved:
                    filename = (
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i)}.csv"
                    )
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    print(i,  jobs[i].last_status)
                elif jobs[i].last_status in ["ERROR", "CANCELLED"]:
                    print(i,  jobs[i].last_status)
        for i in range(N_JOBS):
            if jobs[i].last_status not in ["ERROR", "CANCELLED", "DONE"]:
                ndone = True

    experiments_cleen_up(job_list_path)

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
