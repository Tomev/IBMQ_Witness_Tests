import sys
import time
import matplotlib.pyplot as plt
import datetime

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2,FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from jobobj import  PB
from qiskit.primitives import StatevectorSampler

from settingscloudicm import *

def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    #qubits_list=[[0, 1, 2]]
    qubits_list=[[0,1,2,3,4],[11,12,13,14,15],[20,21,22,23,24],[31,32,33,34,35],[40,41,42,43,44],[51,52,53,54,55],[60,61,62,63,64],[71,72,73,74,75],[80,81,82,83,84],[91,92,93,94,95],\
                 [100,101,102,103,104],[111,112,113,114,115],[120,121,122,123,124],[131,132,133,134,135],[140,141,142,143,144],[151,152,153,154,155]]
    for _ in range(N_JOBS):
        job = PB()
        job.n_repetitions = N_REPETITIONS
        job.add_witness_circuits(qubits_list)
        jobs.append(job)
    
    i = 0
    tok,crn,nm=tcset[0]
    service = QiskitRuntimeService(channel="ibm_cloud",instance=crn,token=tok)
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list_ki_poly.csv"
    
    while i < N_JOBS:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Starting service")
        #tok,crn,nm=tcset2[i//2]
        #service = QiskitRuntimeService(channel="ibm_cloud",instance=crn,token=tok)    
        print(i,nm)
        t = time.localtime()
        backend = service.backend('ibm_kingston', use_fractional_gates=True)
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
        #fig=isa_cir[8].draw("mpl")
        #fig.savefig("zz0.pdf")
        #backend=aer_sim
        sampler = Sampler(backend)
        try:
            jobs[i].queued_job = sampler.run(isa_cir, shots=N_SHOTS)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print("job queued",current_time)
            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                "pars": jobs[i].indices_list,
                "token_id": nm
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
    print("Jobs queued")
    ndone = True
    #return
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
