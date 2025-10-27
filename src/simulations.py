"""
This module is the basis for the jobs simulation.
"""

from multiprocessing import Pool
from typing import List, Optional, Tuple

import psutil
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from tqdm import tqdm

from src.job import PB, Job
from src.utils import *

# Create folder for results.
RESULTS_DIR: str = "polygamy_results"


def prepare_jobs(backend_name: str) -> List[Job]:
    jobs: List[Job] = []

    # Prepare circuits. This is the part to modify.
    if backend_name != "noiseless_simulator":
        print("\tGet target device for Polygamy qubit groups extraction...")

        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ.get("IBMQ_TOKEN_AB_PAID"),
            instance=os.environ.get("IBMQ_CRN_AB_PAID"),
        )

        backend: IBMBackend = service.backend(backend_name, use_fractional_gates=True)

        print("\tExtracting Polygamy qubit groups...")
        qubits: List[Tuple[int]] = find_polygamy_groups(backend)
    else:
        qubits: List[Tuple[int]] = [(0, 1, 2, 3, 4)]

    print("\tPreparing Polygamy jobs...")
    for q_list in qubits:
        job: PB = PB()
        job.add_test_circuits([q_list])
        jobs.append(job)

    return jobs


def simulate_job(job: Job, backend: Optional[IBMBackend] = None) -> str:
    # simulator: AerSimulator = AerSimulator(device="GPU")  # May work sometime.
    simulator: AerSimulator = AerSimulator()

    if backend:
        simulator = simulator.from_backend(backend)
        backend_name = backend.name
    else:
        backend_name: str = "noiseless_simulator"

    sampler: Sampler = Sampler(mode=simulator)

    n_shots: int = 60000
    results_file: str = f"{backend_name}_{job.qubits_list[0]}.csv"
    results_path: str = os.path.join(RESULTS_DIR, results_file)

    print(f"\t\tRunning the job...")
    job.queued_job = sampler.run(job.circuits, shots=n_shots)

    print(f"\t\tSaving job to {results_path}...")
    job.save_to_csv(results_path)

    del job


def simulate_jobs(jobs: List[Job], backend_name: str = "noiseless_simulator") -> None:
    print(f"\tPreparing {backend_name} simulator...")

    # If backend is not specified, we use the noiseless simulator.
    backend: Optional[IBMBackend] = None

    if backend_name != "noiseless_simulator":
        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ.get("IBMQ_TOKEN_AB_PAID"),
            instance=os.environ.get("IBMQ_CRN_AB_PAID"),
        )

        backend: IBMBackend = service.backend(backend_name, use_fractional_gates=True)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"\tRunning circuits transpilation...")
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)

    for job in tqdm(jobs):
        for i in range(len(job.circuits)):
            job.circuits[i] = pm.run(job.circuits[i])

    print("\tRunning the circuits...\n")

    with Pool() as pool:
        pool.starmap(simulate_job, [(job, backend) for job in jobs])

    # TR: Sequential execution.
    """        
    for job in tqdm(jobs):
        simulate_job(job, backend)
        
        # print(f"\n\t{backend_name} job {j}.")

        print(f"\t\tRunning the job...")
        job.queued_job = sampler.run(job.circuits, shots=n_shots)
        results_csv = f"{backend_name}_{str(job.qubits_list[0])}.csv"
        print(f"\t\tRunning saving job...")
        job.save_to_csv(results_csv)
        del job
    """


def main() -> None:
    backends = ["ibm_torino", "ibm_kingston"]
    # backends = ["noiseless_simulator"]

    for backend in backends:
        simulate_jobs(prepare_jobs(backend), backend)


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
