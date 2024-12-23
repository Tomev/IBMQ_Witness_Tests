"""
A script with an example of Leggett-Garg inequality (LGI) triplets finder.
"""

from src.utils import find_lgi_triplets
from src.settings import TOKENS, TOKEN_VARIABLES
from qiskit_ibm_runtime import QiskitRuntimeService

def main():

    print("\tGetting backend...")
    backend_name = "ibm_sherbrooke"

    # Get backend
    service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[1]],
        )
    backend = service.get_backend(backend_name)

    print("\tFinding LGI triplets...\n")
    lgi_triplets = find_lgi_triplets(backend)
    for t in lgi_triplets:
        print(f"\t{t}")

if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment done.")