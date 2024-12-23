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
from tqdm import tqdm

from src.job import LG
from src.utils import *


def simulate_jobs():
    # Prepare circuits.
    job = LG()
    qubits = [[1, 0, 2]]

    epp: float = 0.1

    job.add_test_circuits(qubits, epp)

    # Prepare simulator.
    noisy: bool = False
    print(f"\tPreparing (noisy={noisy}) simulator...")

    backend = AerSimulator()

    if noisy:
        backend_str: str = "ibm_brisbane"

        print(f"\tTargeting {backend_str}...")

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[0]],
        )        

        target_backend = service.get_backend(backend_str)
        backend = backend.from_backend(target_backend)

    print("\tRunning the circuits...\n")
    sampler = Sampler(backend=backend)

    for i, circuit in enumerate(job.circuits):
        result = sampler.run([circuit], shots=1000).result()
        data_pub = result[0].data 

        # cr0 is the name of classical register we use
        counts = data_pub.cr0.get_counts()
        print(f"\t\tJob {i} counts: {counts}")
   

def main():
    print("Start")
    simulate_jobs()
    print("Done")


if __name__ == "__main__":
    main()
