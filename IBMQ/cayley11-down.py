"""
    Module description...
"""

import sys
import time

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2
from qiskit_aer import AerSimulator
from jobsampler import CayleyJob
from settings import *
import json
#from utils import *

#MODULE_FULL_PATH = "/home/jovyan/"
#sys.path.insert(1, MODULE_FULL_PATH)


def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    qubits_list=[2,18,24,30,67,81,73,100,120,126]
    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    #qubits_list = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # TR: For tests

    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list.csv"
    jobs = []
    job_id_table = pd.read_csv(job_list_path, index_col=0)
    for i in range(len(job_id_table)):
        job = CayleyJob()
        job.n_repetitions = N_REPETITIONS
        job.qubits_list=qubits_list
        token=TOKENS["IBMQ_Token_"+job_id_table["token_id"][i]]
        service = QiskitRuntimeService(channel="ibm_quantum",token=token,)  
        job.queued_job = service.job(job_id_table["job_id"][i])
        job.indices_list = json.loads(job_id_table["pars"][i])
        jobs.append(job)
        print(i)
    
    for i in range(len(jobs)):
        try:
            jobs[i].queued_job.cancel()
        except:
            print("Canceled")
        print(jobs[i].queued_job.status())
        #print(jobs[i].queued_job.metrics()["usage"]["quantum_seconds"])
        continue
        #print(jobs[i].queued_job.queue_info())   
        if jobs[i].queued_job.done():
            filename = f"{RESULTS_FOLDER_NAME}/results_tests_{str(i)}.csv"
            jobs[i].save_to_file(filename, ZIP_FILE_NAME)
    #experiments_cleen_up(job_list_path)

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
