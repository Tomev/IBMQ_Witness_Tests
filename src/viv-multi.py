import datetime
import json
import time
import os
import pandas as pd
from zipfile import ZipFile

from qiskit import *

from Jobmulti import save_to_log, WitnessJob

#TOKENS = {
#    'token_name': 'token'
#}

#IBMQ.save_account(TOKENS['token_name'], overwrite=True)

#provider = IBMQ.load_account()
if_researcher_account = None
try:
    researcher_provider = IBMQ.providers()[1]
    if_researcher_account = True
except:
    print("Researcher provider is not available")
    if_researcher_account = False

backends = {
    # "armonk": provider.get_backend('ibmq_armonk'),  # Raised errors
    # "melbourne": provider.get_backend('ibmq_16_melbourne'),  # Raised errors
    #"belem": provider.get_backend('ibmq_belem'),
    #"lima": provider.get_backend('ibmq_lima'),
    #"simulator": BasicAer.get_backend('ibmq_qasm_simulator')
}

# Researcher backends
#if if_researcher_account:
#    backends["perth"] = researcher_provider.get_backend('ibm_perth')
    #backends["nairobi"] = researcher_provider.get_backend('ibm_nairobi')

# -------------------------------- Zmienne sterujące ---------------------------------
brisbane = provider.get_backend('ibm_brisbane')
backend = brisbane
#provider.get_backend('ibmq_qasm_simulator')


N_TEST_CIRCUITS = 0  # ile układów testujących błąd pomiaru, na początku joba
# N_REPETITIONS = backend.configuration().max_experiments // 20  # ile razy powtarzamy każdy kąt w danym jobie
N_REPETITIONS = 10 # max_experiments nie działa dla symulatora
N_JOBS = 1  # ile jobów wykonujemy
#N_SHOTS = backend.configuration().max_shots # ile shotów dla każdego układu
N_SHOTS = 10000
WAIT_TIME = 60  # czas, w sekundach, oczekiwania na ponowną próbę dodania joba do kolejki po niepowodzeniu
SHOULD_RANDOMIZE = True  # randomizacja kątów włączona/wyłączona
RANDOMIZACJA_PAKIETOWA = False
AUTHOR_INITIALS = "TB"
DESCRIPTION = "Tutaj można umieścić krótki opis joba."

log_filename = "log.txt"  # nazwa pliku, w którym będzie zapisywany log
# results_folder_name = "results/wyniki"  # nazwa/ścieżka folderu w którym będą zapisywane wyniki, trzeba go najpierw utworzyć
results_folder_name = "wyniki"
# zip_filename = "results/test_downloading_jobs"  # nazwa pliku zip z wynikami (bez rozszerzenia)
zip_filename = "wyniki-wit-multi"


# -------------------------------- Zmienne sterujące ---------------------------------
# ------------------------------------- Funkcje --------------------------------------
# Generowanie pliku json z danymi i zapakowanie go do zipa
def generate_json_file():
    json_data = {
        "author": AUTHOR_INITIALS,
        "jobs": N_JOBS,
        "tests circuits": N_TEST_CIRCUITS,
        "repetitions": N_REPETITIONS,
        "shots": N_SHOTS,
        "backend": str(backend),
        "randomization": SHOULD_RANDOMIZE,
        "script version": "skrypt_v3_TR",
        "start time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": DESCRIPTION
    }

    json_file_path = results_folder_name + "/data.json"

    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file)

    with ZipFile(zip_filename + '.zip', 'a') as plik_zip:
        plik_zip.write(json_file_path, arcname='wyniki/data.json')
    try:
        os.remove(json_file_path)
    except:
        save_to_log(log_filename, "Error removing json file")


# ------------------------------------- Funkcje --------------------------------------

def main():
    save_to_log(log_filename, "Starting...")  # właściwe rozpoczęcie działania skryptu
    generate_json_file()

    # -------------------------------- Budowanie układów ---------------------------------
    jobs = []
    job_list_table = pd.DataFrame()
    lis=[0,13,24,31,47,69,87,113,118,126]
    for i in range(N_JOBS):
        job = WitnessJob()
        job.n_repetitions = N_REPETITIONS
        # job.add_test_circuits(N_TEST_CIRCUITS)
        job.add_witness_circuits(lis)
        jobs.append(job)
    # -------------------------------- Budowanie układów ---------------------------------

    save_to_log(log_filename, "Circuits built")
    save_to_log(log_filename, "Adding jobs to queue...")

    # ---------------------------- Ustawianie jobow w kolejce ----------------------------
    i = 0
    job_list_path = f'{results_folder_name}/job_list.csv'
    
    while i < N_JOBS:
        try:
            jobs[i].queued_job = execute(jobs[i].circuits, backend=backend, shots=N_SHOTS, optimization_level=0)
            
            # Dodawanie danych joba do tabeli
            job_data = {'job_id': jobs[i].queued_job.job_id(), 
                        'pars': jobs[i].indices_list}
            job_list_table = job_list_table.append(job_data, ignore_index=True)
            job_list_table.to_csv(job_list_path)
        except Exception as alert:
            print(alert)
            save_to_log(log_filename, "Error adding " + str(current_job_index) + ". job to queue. Waiting for the next try...")
            time.sleep(WAIT_TIME)
        ndone=True
        while ndone:
            time.sleep(WAIT_TIME)
            if jobs[i].update_status():
                if jobs[i].last_status == 'DONE' and not jobs[i].if_saved:
                    save_to_log(log_filename, f'{str(i)}. job status: DONE. Saving results to file...')
                    filename = f'{results_folder_name}/wyniki_testy_{str(i)}.csv'
                    jobs[i].save_to_file(filename, zip_filename)
                    i+=1
                    ndone=False
                elif jobs[i].last_status == 'ERROR' or jobs[i].last_status == 'CANCELLED':
                    save_to_log(log_filename, f'{str(i)}. job status: {jobs[i].last_status}')
                    i+=1
                    ndone=False
                    
    # Usuwanie pliku z listą jobów
    with ZipFile(zip_filename + '.zip', 'a') as plik_zip:
        plik_zip.write(job_list_path, arcname='wyniki/job_list.csv')
        
    try:  
        os.remove(job_list_path)
    except:
        save_to_log(log_filename, f'Error removing {job_list_path}')



    
    save_to_log(log_filename, "Program ended.")
    


if __name__ == "__main__":
    main()
