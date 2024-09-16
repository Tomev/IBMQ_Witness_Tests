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
from jobsampler import tele
from qiskit.primitives import StatevectorSampler
 

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
    qubits_list=[[0,1,2]]
    qubits_dir=[[1,1]]
    for _ in range(N_JOBS):
        job = tele()
        job.n_repetitions = N_REPETITIONS
        job.add_witness_circuits(qubits_list,qubits_dir)
        jobs.append(job)
    
    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list_tele.csv"
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
        backend = service.get_backend('ibm_brisbane')
        #backend = service.get_backend('ibm_kyiv')
        #backend = service.get_backend("ibmq_qasm_simulator")
        #backend = GenericBackendV2(num_qubits=127)  # TR: For tests
        
        aer_sim = AerSimulator()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager",current_time)
        pm = generate_preset_pass_manager(backend=aer_sim,optimization_level=0)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager done",current_time)
        isa_cir=pm.run(jobs[i].circuits)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("ISA",current_time)
        backend=aer_sim
        #backend = FakeBrisbane()
        sampler = Sampler(backend=backend)
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
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i)}.csv"
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
