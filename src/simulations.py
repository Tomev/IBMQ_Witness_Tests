"""
    This module is the basis for the jobs simulation.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import IBMBackend
from tqdm import tqdm

from src.job import LG, Job
from src.utils import *

from typing import List

import pickle


def prepare_jobs(backend_name: str) -> List[Job]:
    jobs: List[Job] = []

    # Prepare circuits. This is the part to modify.
    

    service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[1]],
        )

    backend: IBMBackend = service.get_backend(backend_name)

    qubits: List[List[int]] = [[v['x'], v['a'], v['b']] for v in find_lgi_triplets(backend)]

    print(qubits)

    epp: float = 0.1

    for q_list in qubits:
        job: LG = LG()
        job.add_test_circuits([q_list], epp)
        jobs.append(job)

    return jobs


def simulate_jobs(jobs: List[Job], backend_name: str = "") -> None:
    print(f"\tPreparing {backend_name} simulator...")

    simulator: AerSimulator = AerSimulator()

    if backend_name != "":
        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[1]],
        )

        backend: IBMBackend = service.get_backend(backend_name)
        simulator = simulator.from_backend(backend)

    print("\tRunning the circuits...\n")
    sampler = Sampler(backend=simulator)

    for j, job in enumerate(jobs):

        print(f"\t{backend_name} job {j}.")
        results = []

        for i, circuit in enumerate(job.circuits):

            # print(circuit)

            # result = sampler.run([circuit], shots=100).result()
            result = sampler.run([circuit], shots=15000000).result()
            data_pub = result[0].data

            # cr0 is the name of classical register we use
            counts = data_pub.cr0.get_counts()
            print(f"\t\tCircuit {i} counts: {counts}")
            results.append(counts)

        with open(f"{backend_name}_job_{j}.pkl", "wb") as f:
            pickle.dump(results, f) 

 
def main() -> None:
    backends = ["ibm_sherbrooke", "ibm_kyiv", "ibm_brisbane"]

    for backend in backends:
        jobs: List[Job] = prepare_jobs(backend)
        simulate_jobs(jobs, backend)


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
