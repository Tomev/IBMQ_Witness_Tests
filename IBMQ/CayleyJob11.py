from qiskit import *
from typing import List, Dict
from qiskit.circuit import Parameter
import numpy as np
from numpy import pi
import random
import datetime
from zipfile import ZipFile
import pandas as pd
import os


# Save message to log and print it int the console
def save_to_log(filename: str, text: str) -> None:
    data = datetime.datetime.now()
    message = f"{data.strftime('%Y-%m-%d %H:%M:%S')}: {text}"

    log = open(filename, "a")
    log.write(message + '\n')
    log.close()

    print(message)

class CayleyJob:
    def __init__(self) -> None:
        self.n_repetitions = 1
        self.indices_list = []
        self.listvert=[]
        self.circuits = []
        
        self.parameters_list = []
        self.status = None
        self.queued_job = None
        self.last_status = None
        self.test_circuits_number = None
        self.if_saved = False
        
    def _generate_indices_list(self) -> None:
        # i - parameter index in angles_list
        # j - number of SX gates
        for v in self.listvert:
            self.va=[]
            for n in range(self.n_repetitions):           
                for i in range(11):
                    self.va.append(i)
            random.shuffle(self.va)
            #print(*self.va)
            self.indices_list.append(self.va)

    def add_cayley_circuits(self,listvert) -> None:
        self.listvert=listvert
        self._generate_indices_list()
        #print(len(self.indices_list))
        self.circuits.clear()
        #qubit = 0
        #circuit = QuantumCircuit(127, 10)
        for s in range(11*self.n_repetitions):
            self.circuits.append(QuantumCircuit(127, len(listvert)))
            cbit=0
            q=0
            # TB: Usunąłem odejmowanie poprzedniego kąta, wtedy symulator daje poprawne wyniki
            for qubit in listvert:
                
                self.circuits[-1].sx(qubit)
                self.circuits[-1].sx(qubit)
                #print(q,s)
                qs=self.indices_list[q][s]
                if qs<6:
                    for j in range(qs):
                        self.circuits[-1].sx(qubit)
                else:
                    self.circuits[-1].rz(np.pi/2,qubit)
                    self.circuits[-1].sx(qubit)
                    self.circuits[-1].rz(-np.pi/2,qubit)
                    for j in range(qs-6):
                        self.circuits[-1].sx(qubit)
                self.circuits[-1].rz(np.pi/4,qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].rz(-np.pi/4,qubit)
                self.circuits[-1].measure(qubit,cbit)
                cbit+=1
                q+=1
    def update_status(self) -> bool:
        status_before_update = self.last_status
        self.last_status = self.queued_job.status().name
        
        if_changed = None
        if self.last_status == status_before_update:
            if_changed = False
        else:
            if_changed = True
        return if_changed
        
    def save_to_file(self, csv_path: str, zip_filename: str) -> None:     
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        print(self.n_repetitions)
        listvert=self.listvert
        for s in range(11*self.n_repetitions):
            iva=0
            for q in range(len(listvert)):
                iva+=11**q*self.indices_list[q][s]
            indices_i.append(iva)
        print(indices_i)
        pandas_table["i"] = indices_i
        
        # Saving to file
        pandas_table.to_csv(csv_path)
        csv_filename = csv_path.split('/')[-1]
        with ZipFile(zip_filename + '.zip', 'a') as plik_zip:
            plik_zip.write(csv_path, arcname='wyniki/' + csv_filename)

        self.if_saved = True

        try:
            os.remove(csv_path)
        except:
            save_to_log(self.log_filename, f'Error removing {csv_path}')