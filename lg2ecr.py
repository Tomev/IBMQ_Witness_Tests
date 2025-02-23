"""
    Module description...
"""

import sys
import time
import matplotlib.pyplot as plt
import datetime

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2,FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.compiler import transpile, schedule
import qiskit.pulse as pulse
from joblgy import LG
from qiskit.primitives import StatevectorSampler
 

from settings import *
#from utils import *



def run_scripts():
    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    qubits_list=[[53, 60, 41], [114, 115, 113], [0, 1, 14], [68, 69, 55], [26, 16, 25], [4, 3, 15], [103, 102, 104], [82, 83, 81], [54, 45, 64], [33, 20, 39]]
    #[[49, 50, 48], [107, 106, 108], [119, 118, 120], [45, 46, 44], [69, 68, 70], [59, 60, 58], [89, 88, 74], [114, 109, 115], [26, 25, 27], [80, 79, 81]]
    #br[[94, 95, 90], [6, 5, 7], [58, 71, 59], [62, 61, 63], [52, 37, 56], [116, 115, 117], [50, 49, 51], [21, 20, 22], [125, 124, 126], [108, 112, 107]]
    for _ in range(N_JOBS):
        job = LG()
        job.n_repetitions = N_REPETITIONS
        job.add_test_circuits(qubits_list,0.1)
        jobs.append(job)
    
    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list_lgyn_ky_3.csv"
    while i < N_JOBS:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Starting service")
        if i%2==0:
            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=TOKENS[TOKEN_VARIABLES[i//2]],# len(TOKEN_VARIABLES)]],
            )
        print(i)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("service started",current_time)
        print(service.active_account())
        #backend = service.backend('ibm_sherbrooke')
        backend = service.backend('ibm_kyiv')
        #backend = service.get_backend("ibmq_qasm_simulator")
        #backend = GenericBackendV2(num_qubits=127)  # TR: For tests
        #backend = FakeSherbrooke()
        aer_sim = AerSimulator()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager",current_time)
        pm = generate_preset_pass_manager(backend=backend,optimization_level=0)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager done",current_time)
        isa_cir=pm.run(jobs[i].circuits)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("ISA",current_time)
        #backend=aer_sim
        sampler = Sampler(backend)
        #sampler = StatevectorSampler()
        #print(backend)
        try:
            jobs[i].queued_job = sampler.run(isa_cir, shots=N_SHOTS)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print("job queued",current_time)
            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                "pars": jobs[i].indices_list,
                "token_id": TOKEN_VARIABLES[i//2].split("_")[-1]
            }
            job_list_table = pd.concat(
                [job_list_table, pd.DataFrame([job_data])], ignore_index=True
            )
            job_list_table.to_csv(job_list_path)
            t = time.localtime()
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
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i+24)}.csv"
                    )
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    print(i,  jobs[i].last_status)
                elif jobs[i].last_status in ["ERROR", "CANCELLED"]:
                    print(i,  jobs[i].last_status)
                    print(jobs[i].queued_job.error_message())
                    print(jobs[i].queued_job.metrics()["usage"]["quantum_seconds"])
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
