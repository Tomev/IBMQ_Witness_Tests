"""
    Module description...
"""

import sys
import time

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2
from qiskit_aer import AerSimulator
from jobsampler import BellJob
from settings import *
from directionextractor import *
import json
#from utils import *

#MODULE_FULL_PATH = "/home/jovyan/"
#sys.path.insert(1, MODULE_FULL_PATH)


def run_scripts():
    print(len(TOKENS))
    print(len(TOKEN_VARIABLES))
    for i in range(len(TOKENS)):
        try:
            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=TOKENS[TOKEN_VARIABLES[i]],
                )
            print(service.active_account())
        except Exception as alert:
            print(alert)
            print(TOKEN_VARIABLES[i])
        print(i)
        continue
        extr=DirectionExtractor('ibm_sherbrooke',token=TOKENS[TOKEN_VARIABLES[i]])
        a,b=extr.extract_directions()
        print('[',end='')
        for qubit in [3,28,40,66,78,104,116]:
            qubits=[j for j in range(qubit-3,qubit+4)]
            h= len(qubits)-1
            di=[]

            for j in range(h):
                if "ecr"+str(qubits[j])+"_"+str(qubits[j+1]) in a:
                    di.append(0)
                elif "ecr"+str(qubits[j+1])+"_"+str(qubits[j]) in a:
                    di.append(1)
                else:
                    di.append(2)
            print(di,end=',')
        print(']') 
        break
    backend = service.get_backend('ibm_osaka')
    print(backend.dt)
    x_pulse = backend.defaults().instruction_schedule_map.get('sx', (0,)).instructions[0][1].pulse
    print(x_pulse)
    print(backend.defaults().instruction_schedule_map)
    print(backend.default_rep_delay)
    return

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
