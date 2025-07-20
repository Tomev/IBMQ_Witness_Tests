"""
This module contains utulity functions for our jobs.
"""

__author__ = "Tomasz Rybotycki"

import os
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Tuple
from zipfile import ZipFile

from qiskit_ibm_runtime.ibm_backend import IBMBackend
from tqdm import tqdm

from src.settings import *


@dataclass
class ExperimentSetup:
    backend: str
    qubit_group: List[int]


def experiments_clean_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        os.remove(job_list_path)
    except Exception as alert:
        print(alert)


def find_qubit_connections(backend: IBMBackend, qubit: int) -> List[int]:
    """
    Find all qubits that are connected to the given qubit.

    :params:
        backend:     IBM backend.
        qubit:       The qubit to check connections for.

    :return:
        A list of connections in the backend that qubit controls.
    """
    coupling_map = backend.coupling_map

    return [con[1] for con in coupling_map if con[0] == qubit]


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
        connections = find_qubit_connections(backend, x)

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


def find_qubit_neighbors(backend: IBMBackend, qubit: int) -> List[int]:
    """
    Find all qubits that are neighbors of the given qubit.

    :note:
        The neighbors are defined as qubits that are connected to the given
        qubit in the coupling map.

    :params:
        backend:     IBM backend.
        qubit:       The qubit to check neighbors for.

    :return:
        A list of neighbors in the backend that qubit controls.
    """
    coupling_map = backend.coupling_map

    return [con[1] for con in coupling_map if con[0] == qubit]


def find_polygamy_groups(backend: IBMBackend) -> List[List[int]]:
    """
    Brute-force search for all qubits that satisfy connections required
    for the Polygamy experiment.

    For each qubit

    :note:
        Full brute-force won't work, as there is too much combinations to
        check.

    :note:
        The connections are as follows (we disregard the direction of the gate):
        1 - 2 - 3 - 4 - 5

    :params:
        backend:     IBM backend.

    :return:
        A list of lists of qubits that are connected in the way required for
        the Polygamy experiment.
    """
    polygamy_groups: List[Tuple[int]] = []
    group_len: int = 5

    n_qubits = backend.configuration().n_qubits

    # We will generate possible groups and the filter them. For each qubit
    # we will assume it's the central one (qubit 3 in the scheme above).
    for qubit in tqdm(range(n_qubits)):
        # Find all qubits that are neighbors of the current qubit.
        neighbors = find_qubit_neighbors(backend, qubit)

        # Central qubit needs at least 2 neighbors to form a group.
        if len(neighbors) < 2:
            continue

        # Generate all combinations of neighbors of the current qubit.
        for group in combinations(neighbors, 2):
            # Now we need second level neighbors of the group.
            # For qubit 2 and qubit 4 of the neighbors.
            q2_neighbors: List[int] = find_qubit_neighbors(backend, group[0])
            q4_neighbors: List[int] = find_qubit_neighbors(backend, group[1])

            if len(q2_neighbors) < 1 or len(q4_neighbors) < 1:
                continue

            for l2_neighbor in product(q2_neighbors, q4_neighbors):
                polygamy_groups.append(
                    (l2_neighbor[0], group[0], qubit, group[1], l2_neighbor[1])
                )

    # Filter groups.
    polygamy_groups = [g for g in polygamy_groups if len(set(g)) == group_len]

    # for g in polygamy_groups:
    #    print(g)

    return polygamy_groups
