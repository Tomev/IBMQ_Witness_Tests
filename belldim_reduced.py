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
from settings import *
#from utils import *

#MODULE_FULL_PATH = "/home/jovyan/"
#sys.path.insert(1, MODULE_FULL_PATH)
def title_generator(program, device):
    return ''

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

    i = 8
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list.csv"

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(i)
        print(service.active_account())
        #backend = service.get_backend('ibm_brisbane')
        backend = service.get_backend('ibm_sherbrooke')

        pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
        sampler = Sampler(backend=backend)


        try:
            isa_cir=pm.run(jobs[i].circuits)
            jobs[i].queued_job = sampler.run(isa_cir, shots=N_SHOTS)

            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                "token_id": TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)].split("_")[-1]
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
                #print(jobs[i].last_status)

    experiments_cleen_up(job_list_path)

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
