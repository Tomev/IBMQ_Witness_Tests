"""
    This module contains functionality related to extracting the information
    about 2-qubit gates direction from IBM quantum devices.
"""

__author__ = "Tomasz Rybotycki"

from typing import Any, Dict, List, Optional

#from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService


class DirectionExtractor:
    """
    Extracts directions of the 2-qubit gates from the IBM quantum devices.
    """

    def __init__(self, device_name: str, token: Optional[str] = None) -> None:
        """
        Initializes the `DirectionExtractor` class.

        :param device_name:
            Name of the device from which the gate directions will be
            extracted.
        :param token:
            IBMQ token.
        """

        self.provider: QiskitRuntimeService

        if token is None:
            self.provider = QiskitRuntimeService("ibm_quantum")
        else:
            self.provider = QiskitRuntimeService("ibm_quantum",token=token)

        self.device: BackendV2 = self.provider.get_backend(device_name)

    def extract_directions(self, save_to_file: bool = False) -> Dict[str, List[str]]:
        """
        Finds the flow of the qubits in the device.

        :param save_to_file:
            If True, the results will be saved to the file.

        :return:
            Dictionary with the qubits as keys and lists of qubits that
            they are connected to as values.
        """
        gates: List[str, Dict[str, Any]] = self.device.properties().to_dict()["gates"]
        gates = [gate for gate in gates if len(gate["qubits"]) > 1]

        gate_names = [gate["name"] for gate in gates]
        gate_directions = [
            0 if gate["qubits"][0] > gate["qubits"][1] else 1 for gate in gates
        ]

        if save_to_file:
            with open("directions.txt", "w") as file:
                file.write(f"{str(gate_names)}\n")
                file.write(str(gate_directions))

        return gate_names, gate_directions
