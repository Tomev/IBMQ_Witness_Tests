"""
    Module description...
"""

import sys
import time
import matplotlib.pyplot as plt

import pandas as pd
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeBrisbane,FakeProviderForBackendV2,FakeSherbrooke
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.compiler import transpile, schedule
import qiskit.pulse as pulse
from jobsampler import VivianiJobSwap
from settings import *
#from utils import *

#MODULE_FULL_PATH = "/home/jovyan/"
#sys.path.insert(1, MODULE_FULL_PATH)
def title_generator(program, device):
    return ''

def run_scripts():

    jobs = []
    job_list_table = pd.DataFrame()

    # Job preparation
    qubits_lista=[0,13,6,10,21,24,27,31,37,41,45,49,57,62,65,70,75,78,82,87,94,97,101,105,114,117,121,125]
    qubits_listb=[1,12,5,9,20,25,28,32,38,42,46,50,58,61,66,69,76,79,83,86,95,98,102,106,113,118,122,126]
    qubits_dir=[1,1,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,0,1]
    #qubits_list=[0,3,6,9,13,19,23,27,31,37,40,44,48,51,57,61,65,69,75,78,82,86,89,95,99,103,107,113, 116, 119,123,126]
    #qubits_list=[2,18,24,30,67,81,73,100,120,126]
    # qubits_list = [11, 18, 24, 67, 70, 77, 81, 96, 108, 120]
    #qubits_list = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # TR: For tests

    for _ in range(N_JOBS):
        job = VivianiJobSwap()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(qubits_lista,qubits_listb,qubits_dir)
        jobs.append(job)

    i = 0
    job_list_path = f"{RESULTS_FOLDER_NAME}/job_list.csv"

    while i < N_JOBS:

        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token=TOKENS[TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)]],
        )
        print(i)
        print(service.active_account())
        #backend = service.get_backend('ibm_brisbane')
        backend = service.get_backend('ibm_sherbrooke')
        #backend = service.get_backend("ibmq_qasm_simulator")
        #backend = GenericBackendV2(num_qubits=127)  # TR: For tests
        #backend = FakeSherbrooke()
        #pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
        sampler = Sampler(backend=backend)
        #print(backend)
        #print(len(qubits_lista),len(qubits_listb),len(qubits_dir))
        #for x in range(len(qubits_lista)):
        #    if qubits_dir[x]:
        #        print(qubits_listb[x], "->",qubits_lista[x])
        #    else:
        #        print(qubits_lista[x], "->",qubits_listb[x])
        #sched=schedule(isa_cir[9],backend)
        print("pm - start")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
        print("pm - end")
        isa_cir=pm.run(jobs[i].circuits)
        print("pm - x")
        sched=schedule(isa_cir[9],backend)
        my_style = {
            'formatter.general.fig_width': 20, # Wysokość obrazka
            'formatter.general.fig_chart_height': 20, # Szerokość obrazka
            'formatter.text_size.axis_label': 8.5, # Wielkość tekstu osie
            'formatter.text_size.annotate': 8.5, # Wielkość tekstu opisy impulsów
            'formatter.label_offset.pulse_name': 0.3, # Odstęp opisów impulsów od wykresu
            'formatter.channel_scaling.drive': 1, # Wielkość rysowanych impulsów
            'layout.figure_title': title_generator,
        }
        style = my_style #IQXStandard(**my_style)
        col = 3+3/8
        fig, axs = plt.subplots(1, figsize=(1.5*col, col),dpi=300)
        backend.name = lambda name=backend.name: name # uncomment to see the schedule drawer work
        fig=sched.draw(backend=backend)
        #, style=style, axis=axs, show_waveform_info=False)
        print(fig)
        fig.savefig('waveform_sherbrooke_noswap_viv.pdf', bbox_inches='tight')
        fig
        break
        isa_cir=pm.run(jobs[i].circuits)
        #backend.name = lambda name=backend.name: name # uncomment to see the schedule drawer work
        fig=isa_cir[9].draw(output='mpl',fold=1000)
        #, style=style, axis=axs, show_waveform_info=False)
        #print(fig)
        #fig.savefig('cir_sherbrooke_vivs.pdf', bbox_inches='tight')
        #fig
        #break
        try:
            isa_cir=pm.run(jobs[i].circuits)
            jobs[i].queued_job = sampler.run(isa_cir, shots=N_SHOTS)

            job_data = {
                "job_id": jobs[i].queued_job.job_id(),
                "pars": jobs[i].indices_list,
                "token_id": TOKEN_VARIABLES[i % len(TOKEN_VARIABLES)].split("_")[-1]
            }
            job_list_table = pd.concat(
                [job_list_table, pd.DataFrame([job_data])], ignore_index=True
            )
            job_list_table.to_csv(job_list_path)
            i += 1
        except Exception as alert:
            print(alert)
            time.sleep(WAIT_TIME)
    ndone = True
    return
    while ndone:
        ndone = False
        time.sleep(WAIT_TIME)
        for i in range(N_JOBS):
            if jobs[i].update_status():
                print(i,jobs[i].last_status)
                if jobs[i].last_status == "DONE" and not jobs[i].if_saved:
                    filename = (
                        f"{RESULTS_FOLDER_NAME}/results_tests_{str(i)}.csv"
                    )
                    jobs[i].save_to_file(filename, ZIP_FILE_NAME)
                    print(i,  jobs[i].last_status)
                elif jobs[i].last_status in ["ERROR", "CANCELLED"]:
                    print(i,  jobs[i].last_status)
        for i in range(N_JOBS):
            if jobs[i].last_status not in ["ERROR", "CANCELLED", "DONE"]:
                ndone = True
                #print(jobs[i].last_status)

    experiments_cleen_up(job_list_path)

def main():
    print("Start")
    run_scripts()
    print("Done")


if __name__ == "__main__":
    main()
