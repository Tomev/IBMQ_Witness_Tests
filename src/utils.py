"""
    This module contains utulity functions for our jobs.
"""

__author__ = "Tomasz Rybotycki"

import os
from zipfile import ZipFile
from typing import List, Dict, Iterable
from itertools import combinations
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from tqdm import tqdm

from src.settings import *


def experiments_clean_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        os.remove(job_list_path)
    except Exception as alert:
        print(alert)


def find_lgi_triplets(backend: IBMBackend) -> List[Dict[str, int]]:
    """
    Given the IBM backend, find the qubit triplets for the Laggett-Garg test experiment.
    We require that qubits X, A, B are connected in the following way:
    
    X -> A
     
    and
     
    X -> B,

    where the arrow denotes the direction of the entangling gate.

    :params:
        backend:     IBM backend. 

    :return:
        A list of LGI-eligible qubit triplets.
    """
    lgi_triplets: List[Dict[str, int]] = []

    coupling_map = backend.coupling_map

    def format_connection(con: Iterable[Iterable[int]]):
        return {"x": con[0][0], "a": con[0][1], "b": con[1][1]}

    for x in tqdm(range(backend.configuration().n_qubits)):
        # Find all qubits that x can control.
        connections = [c for c in coupling_map if c[0] == x]

        if len(connections) < 2:
            continue

        if len(connections) > 2:
            # Prepare 2-length permutations of the connections.
            connections = list(combinations(connections, 2))
        if len(connections) == 2:
            connections = [connections]  # Hax for more general processing.

        for con in connections:
            lgi_triplets.append(format_connection(con))

    return lgi_triplets
