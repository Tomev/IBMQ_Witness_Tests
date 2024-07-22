"""
    This module is the basis for the witness experiments.
"""

import time

import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from qiskit_aer import AerSimulator

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from job import Job, VivianiJob
from utils import *

from datetime import timedelta, datetime
from tqdm import tqdm

def enqueue_jobs():

    service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[0]],
        )
    #service = QiskitRuntimeService()

    backend_str: str = "ibm_brisbane"
    backend = service.get_backend(backend_str)
    backend = AerSimulator().from_backend(backend)
    #backend = service.get_backend("ibmq_qasm_simulator")

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    qubits_list = [0, 1]  # TR: For tests

    """
    for _ in range(N_JOBS):
        job: Job = VivianiJob()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(qubits_list)
        jobs.append(job)

    isa_circuits = []

    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    start = datetime.now()
    for job in tqdm(jobs):
        for circuit in job.circuits:
            isa_circuits.append(pass_manager.run(circuit))
    print("Time to transpile: ", timedelta(seconds=(datetime.now() - start).seconds))

    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    start = datetime.now()
    for circuit in tqdm(isa_circuits):
        pass_manager.run(circuit)
    print("Time to transpile ISA: ", timedelta(seconds=(datetime.now() - start).seconds))

    # Return before running the jobs.
    return 0
    """

    i: int = 0

    # TODO TR:  Might be better to do it with database at some point.
    #           Or possibly with the environment variable.
    if os.path.exists(JOB_TRACKER_FILE_NAME):
        with open(JOB_TRACKER_FILE_NAME, "r") as file:
            i = int(file.read())
    else:
        print("No file tracker found.")
        
    print(f"\t{i}")
    return

    job_list_path: str = f"{RESULTS_FOLDER_NAME}/job_list.csv"

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(service.active_account())
        

        try:
            sampler = Sampler(backend=backend)
            jobs[i].queued_job = sampler.run(jobs[i].circuits, shots=N_SHOTS,
            skip_transpilation=True)

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

    experiments_clean_up(job_list_path)

def main():
    print("Start")
    enqueue_jobs()
    print("Done")


if __name__ == "__main__":
    main()
