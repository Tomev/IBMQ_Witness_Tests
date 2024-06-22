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
from jobsampler import BellJob
from settings import *
import json
#from utils import *

#MODULE_FULL_PATH = "/home/jovyan/"
#sys.path.insert(1, MODULE_FULL_PATH)


def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()
    qubits_list=[3,28,40,66,78,104,116]
    qubits_dir=[[1, 1, 1, 0, 1, 0],[1, 1, 0, 1, 0, 0],[1, 1, 0, 1, 0, 0],[1, 0, 1, 0, 1, 0],[1, 0, 1, 0, 0, 0],[0, 1, 1, 0, 1, 1],[0, 1, 0, 0, 0, 1]]
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list.csv"
    jobs = []
    job_id_table = pd.read_csv(job_list_path, index_col=0)
    for i in range(len(job_id_table)):
        job = BellJob()
        job.n_repetitions = N_REPETITIONS
        job.qubits_list=qubits_list
        job.qubits_dir=qubits_dir
        token=TOKENS["IBMQ_Token_"+job_id_table["token_id"][i]]
        service = QiskitRuntimeService(channel="ibm_quantum",token=token,)  
        job.queued_job = service.job(job_id_table["job_id"][i])
        jobs.append(job)
        print(i)
    
    for i in range(len(jobs)):
       # try:
        #    jobs[i].queued_job.cancel()
        #except:
        #    print("Canceled")
        #continue
        print(jobs[i].queued_job.status())
        print(jobs[i].queued_job.error_message())
        print(jobs[i].queued_job.metrics()["usage"]["quantum_seconds"])
        #continue
        print(jobs[i].queued_job.queue_info())
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
