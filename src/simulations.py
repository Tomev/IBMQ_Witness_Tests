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


def prepare_jobs() -> List[Job]:
    jobs: List[Job] = []
    
    # Prepare circuits. This is the part to modify.
    job: LG = LG()
    qubits: List[List[int]] = [[1, 0, 2]]

    epp: float = 0.1

    job.add_test_circuits(qubits, epp)

    jobs.append(job)

    return jobs


def simulate_jobs(jobs: List[Job], noisy: bool = False) -> None:
    print(f"\tPreparing (noisy={noisy}) simulator...")

    simulator: AerSimulator = AerSimulator()

    if noisy:
        backend_str: str = "ibm_brisbane"

        print(f"\tTargeting {backend_str}...")

        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[0]],
        )        

        backend: IBMBackend = service.get_backend(backend_str)
        simulator = simulator.from_backend(backend)

    print("\tRunning the circuits...\n")
    sampler = Sampler(backend=simulator)

    for job in jobs:
        for i, circuit in enumerate(job.circuits):
            result = sampler.run([circuit], shots=1000).result()
            data_pub = result[0].data 

            # cr0 is the name of classical register we use
            counts = data_pub.cr0.get_counts()
            print(f"\t\tJob {i} counts: {counts}")
   

def main() -> None:
    jobs: List[Job] = prepare_jobs()
    simulate_jobs(jobs)
    

if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
