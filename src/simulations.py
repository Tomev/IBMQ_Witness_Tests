"""
This module is the basis for the jobs simulation.
"""

from mp_utils import NonDeamonicPool
from typing import List, Optional, Tuple

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from tqdm import tqdm

from job import VivianiPPCZ, Job
from utils import *

# Create folder for results.
SCHEDULING = "alap"  # "asap" or "alap"
RESULTS_DIR: str = "pp_sim_results"


def prepare_jobs(backend_name: str) -> List[Job]:
    jobs: List[Job] = []

    # Prepare circuits. This is the part to modify.
    if backend_name != "noiseless_simulator":
        print("\tGet target device for PP qubit groups extraction...")

        # TR: Extraction is usually backend-dependent, due to connectivity scheme.
        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ.get("IBMQ_TOKEN_AB_PAID"),
            instance=os.environ.get("IBMQ_CRN_AB_PAID"),
        )

        backend: IBMBackend = service.backend(backend_name, use_fractional_gates=True)
        

        print("\tExtracting PP qubit groups...")
        # qubits: List[Tuple[int]] = get_predefined_torino_pp_groups()
        qubits: List[Tuple[int]] = get_predefined_pittsburgh_pp_groups()
    else:
        qubits: List[Tuple[int]] = [(0, 1, 2)]

    print("\tPreparing PP jobs...")
    for q_list in qubits:
        job: VivianiPPCZ = VivianiPPCZ(n_qubits=backend.num_qubits)
        job.add_witness_circuits([q_list])
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

    n_shots: int = 3200000
    results_file: str = f"{backend_name}_{job.qubits_list[0]}.csv"
    results_path: str = os.path.join(RESULTS_DIR, results_file)

    print(f"\t\tRunning the job...")
    job.queued_job = sampler.run(job.circuits, shots=n_shots)

    print(f"\t\tSaving job to {results_path}...")
    job.save_to_csv(results_path)

    del job

def simulate_job_name(job: Job, backend_name: Optional[IBMBackend] = None) -> str:
    # simulator: AerSimulator = AerSimulator(device="GPU")  # May work sometime.
    simulator: AerSimulator = AerSimulator()
    backend: Optional[IBMBackend] = None

    if backend_name != "noiseless_simulator":
        service: QiskitRuntimeService = QiskitRuntimeService(
            channel="ibm_cloud",
            token=os.environ.get("IBMQ_TOKEN_AB_PAID"),
            instance=os.environ.get("IBMQ_CRN_AB_PAID"),
        )

        backend = service.backend(backend_name, use_fractional_gates=True)


    if backend:
        simulator = simulator.from_backend(backend)
        backend_name = backend.name
    else:
        backend_name: str = "noiseless_simulator"

    sampler: Sampler = Sampler(mode=simulator)

    n_shots: int = 3200000
    results_file: str = f"{backend_name}_{job.qubits_list[0]}_{SCHEDULING}.csv"
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
    # pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend, scheduling_method=SCHEDULING)

    for job in tqdm(jobs):
        for i in range(len(job.circuits)):
            job.circuits[i] = pm.run(job.circuits[i])

    print("\tRunning the circuits...\n")

    n_workers: int = 4
    with NonDeamonicPool(n_workers) as pool:
        # TR: Watch out for RAM!
        # TIP TR: 3 200 000, shots for 8 workers was too much. 4 was good. 6 and 5 ran, but got stuck on saving results.
        pool.starmap(simulate_job_name, [(job, backend_name) for job in jobs])

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
    # backends = ["ibm_torino", "ibm_kingston"]
    backends = ["ibm_pittsburgh"]
    # backends = ["noiseless_simulator"]

    for backend in backends:
        simulate_jobs(prepare_jobs(backend), backend)


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
