"""
    This module is the basis for the jobs simulation.
"""

from typing import List

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from tqdm import tqdm

from src.job import Job, VivianiJob
from src.utils import *


def prepare_jobs(backend_name: str) -> List[Job]:
    jobs: List[Job] = []

    qubits = [i for i in range(127)]

    if backend_name == "noiseless_simulator":
        qubits = [0]

    print("\tPreparing Viviani jobs...")
    for q_list in qubits:
        job: VivianiJob = VivianiJob()
        job.add_witness_circuits([q_list])
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

    n_shots = int(1e6)
    # n_shots = 100

    # zip_file_name = "LG_sim"
    zip_file_name = "viviani_sim"

    # TR TODO: This probably could be parallelized. Figure out how to do it.
    for j, job in tqdm(enumerate(jobs)):

        print(f"\n\t{backend_name} job {j}.")

        print(f"\t\tRunning the job...")
        job.queued_job = sampler.run(job.circuits, shots=n_shots)
        results_csv = f"{backend_name}_{str(job.qubits_list[0])}.csv"
        print(f"\t\tRunning saving job...")
        job.save_to_file(results_csv, zip_file_name)


def main() -> None:
    # backends = ["ibm_brisbane", "ibm_sherbrooke", "ibm_kyiv", "noiseless_simulator"]
    backends = ["noiseless_simulator", "ibm_brisbane"]

    for backend in backends:
        jobs: List[Job] = prepare_jobs(backend)
        simulate_jobs(jobs, backend)


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
