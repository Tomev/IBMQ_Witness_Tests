"""
    This module is the basis for the jobs simulation.
"""

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import IBMBackend
from tqdm import tqdm

from src.job import LG, Job
from src.utils import *

from typing import List


def prepare_jobs(backend_name: str) -> List[Job]:
    jobs: List[Job] = []

    # Prepare circuits. This is the part to modify.
    if backend_name != "noiseless_simulator":
        print("\tGet target device for LG qubit trilplets extraction...")
        service: QiskitRuntimeService = QiskitRuntimeService(
                channel="ibm_quantum",
                token=TOKENS[TOKEN_VARIABLES[1]],
            )

        backend: IBMBackend = service.get_backend(backend_name)

        print("\tExtracting LG qubit triplets...")
        qubits: List[List[int]] = [[v['x'], v['a'], v['b']] for v in find_lgi_triplets(backend)]
    else:
        qubits: List[List[int]] = [[0, 1, 2]]

    # print(qubits)

    epp: float = 0.1

    print("\tPreparing LG jobs...")
    for q_list in qubits:
        job: LG = LG()
        job.add_test_circuits([q_list], epp)
        jobs.append(job)

    return jobs


def simulate_jobs(jobs: List[Job], backend_name: str = "noiseless_simulator") -> None:
    print(f"\tPreparing {backend_name} simulator...")

    simulator: AerSimulator = AerSimulator()

    if backend_name != "noiseless_simulator":
        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[1]],
        )

        backend: IBMBackend = service.get_backend(backend_name)
        simulator = simulator.from_backend(backend)

    print("\tRunning the circuits...\n")
    sampler = Sampler(backend=simulator)

    n_shots = 15000000
    # n_shots = 100
    
    zip_file_name = "LG_sim"

    # TR TODO: This probably could be parallelized. Figure out how to do it.
    for j, job in tqdm(enumerate(jobs)):

        print(f"\n\t{backend_name} job {j}.")

        print(f"\t\tRunning the job...")
        job.queued_job = sampler.run(job.circuits, shots=n_shots)
        results_csv = f"{backend_name}_{str(job.qubits_list[0])}.csv"
        print(f"\t\tRunning saving job...")
        job.save_to_file(results_csv, zip_file_name)

 
def main() -> None:
    backends = ["ibm_brisbane", "ibm_sherbrooke", "ibm_kyiv", "noiseless_simulator"]

    for backend in backends:
        jobs: List[Job] = prepare_jobs(backend)
        simulate_jobs(jobs, backend)


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
