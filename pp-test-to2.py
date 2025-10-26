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
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2,FakeSherbrooke, FakeTorino
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from jobsampler import VivianiPPCZ
from qiskit.primitives import StatevectorSampler
 

from settingscloud import *
#from utils import *



def run_scripts():
    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    qubits_list=[[0,1,2],[4,5,6],[8,9,10],[12,13,14],[20,21,22],[25,35,44],[27,28,29],[31,32,33],\
            [38,53,57],[40,41,42],[46,47,48],[50,56,69],[59,72,78],[61,62,63],[65,66,67],[71,75,90],\
            [76,91,95],[80,81,82],[84,85,86],[88,94,107],[97,98,99],[101,111,120],[103,104,105],[109,113,128],\
            [129,114,115],[117,118,130],[131,122,123],[125,126,132]]
    #qubits_list=[[129,114,115]]
    qt=[0 for i in range(133)]
    for ql in qubits_list:
        for qu in ql:
            qt[qu]=qt[qu]+1
            if qt[qu]>1:
                print(qu)
                return
            
    for _ in range(N_JOBS):
        job = VivianiPPCZ()
        job.n_repetitions = N_REPETITIONS
        job.add_witness_circuits(qubits_list)
        jobs.append(job)
    
    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list_pp_to_3_as_for.csv"
    while i < N_JOBS:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Starting service")
        tok,crn,nm=tcset8[i//2]
        service = QiskitRuntimeService(channel="ibm_cloud",instance=crn,token=tok)    
        print(i,nm)
        t = time.localtime()
        backend = service.backend('ibm_torino')
        #backend = service.get_backend('ibm_kyiv')
        #backend = service.get_backend("ibmq_qasm_simulator")
        #backend = GenericBackendV2(num_qubits=127)  # TR: For tests
        #backend = FakeSherbrooke()
        aer_sim = AerSimulator()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager",current_time)
        pm = generate_preset_pass_manager(backend=backend,optimization_level=0,scheduling_method='asap')
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("pass manager done",current_time)
        isa_cir=pm.run(jobs[i].circuits)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("ISA",current_time)
        #backend=FakeTorino()
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
    ndone = True

    while ndone:
        ndone = False
        time.sleep(WAIT_TIME)
        for i in range(N_JOBS):
            if jobs[i].update_status():
                print(i,jobs[i].last_status)
                if jobs[i].last_status == "DONE" and not jobs[i].if_saved:
                    filename = (
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i+120)}.csv"
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
