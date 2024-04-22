from qiskit import *
from typing import List, Dict
import numpy as np
from numpy import pi
import random
import datetime
from zipfile import ZipFile
import pandas as pd
import os


# zapis danych do logu i wyświetlenie komunikatu
def save_to_log(filename: str, text: str) -> None:
    data = datetime.datetime.now()
    message = f"{data.strftime('%Y-%m-%d %H:%M:%S')}: {text}"

    log = open(filename, "a")
    log.write(message + '\n')
    log.close()

    print(message)

class Job:

    # TODO TR: Specify the types, just as in circuits.
    parameters_list = []
    circuits: List[QuantumCircuit] = []
    last_status = None
    queued_job = None
    status = None
    test_circuits_number = None
    if_saved: bool = False

    def __init__(self, log_filename="log.txt"):
        self.parameters_list = []
        self.circuits = []
        self.status = None
        self.queued_job = None
        self.last_status = None
        self.test_circuits_number = None
        self.if_saved = False
        self.log_filename = log_filename

    def add_test_circuits(self, test_number: int) -> None:
        self.test_circuits_number = test_number
        # test stanu 0 quibtu
        circuit_test_0 = QuantumCircuit(1, 1)
        circuit_test_0.measure(0, 0)
        # test stanu 1 quibtu
        circuit_test_1 = QuantumCircuit(1, 1)
        circuit_test_1.x(0)
        circuit_test_1.measure(0, 0)
        # dodanie ukladow testowych do
        for j in range(0, test_number):
            self.circuits.append(circuit_test_0)
            self.circuits.append(circuit_test_1)


    def update_status(self):
        status_before_update = self.last_status
        self.last_status = self.queued_job.status().name

        if_changed = None
        if self.last_status == status_before_update:
            if_changed = False
        else:
            if_changed = True
        return if_changed

    # Zapis danych do pliku
    def save_to_file(self, csv_path, zip_filename):
        wyniki = self.queued_job.result().get_counts()
        tabela = pd.DataFrame.from_dict(wyniki).fillna(0)

        theta = []
        # część testowa
        # for i in range(0, 2 * self.test_circuits_number):
        #   theta.append("TEST")
        # kąty w odpowiedniej kolejności
        # theta.extend(self.parameters_list)
        for n in range(101):
            theta.append(f"Test_Circuit_n_{n}")

        print(tabela)

        # dodanie właściwej kolumny do danych
        tabela["theta"] = theta
        tabela.to_csv(csv_path)

        csv_filename = csv_path.split('/')[-1]
        with ZipFile(zip_filename + '.zip', 'a') as plik_zip:
            plik_zip.write(csv_path, arcname='wyniki/' + csv_filename)

        self.if_saved = True

        try:
            os.remove(csv_path)
        except:
            save_to_log(self.log_filename, f'Error removing {csv_path}')


class RotationJob(Job):

    def __init__(self):
        Job.__init__()
        self.theta_range = np.linspace(-np.pi, 7 * np.pi / 8, 16)


    def add_rotation_circuits(self, parameters_list: List[float]) -> None:
        self.parameters_list = parameters_list

        theta = Parameter('t')
        circuit_rotation = QuantumCircuit(1, 1)
        circuit_rotation.sx(0)
        circuit_rotation.rz(theta + np.pi, 0)
        circuit_rotation.sx(0)
        circuit_rotation.measure(0, 0)

        for t in self.parameters_list:
            self.circuits.append(circuit_rotation.bind_parameters({theta: t}))

    def add_test_circuits_with_rotations(self) -> None:
        """
            Prepares test circuits, as described in 22 VI e-mail, by prof. AB. In each
            circuit we start with |0> and then apply rz(theta) and X gate n number of
            times where n = 0, ..., 100. We expect to run everything on Lima in one job.
            The results depend on the number of iterations.
        """
        for n in range(101):
            test_circuit = QuantumCircuit(1, 1)
            for _ in range(n):
                test_circuit.rz(random.choice(self.theta_range), 0)
                test_circuit.x(0)
            test_circuit.measure(0, 0)
            test_circuit.name = f"Test_Circuit_n_{n}"
            self.circuits.append(test_circuit)



class WitnessJob(Job):
    # TB: Poprawiłem wartości kątów
    def __init__(self) -> None:
        super().__init__()
        
        #self.alphas =   [0,     2 * pi / 3, 2 * pi / 3, -2 * pi / 3,    -2 * pi / 3]
        #self.betas =    [0,     pi / 6,     -pi / 6,    pi / 6,         -pi / 6]
        #self.thetas =   [2 * pi / 3 + pi,    2 * pi / 3 + pi,  -2 * pi / 3 + pi,   -2 * pi / 3 + pi]
        #self.phis =     [pi / 6 + pi,        -pi / 6 + pi,     pi / 6 + pi,       -pi / 6 + pi]

        # Optimized angles
        eta = 1.23095942
        #self.alphas =   [0,     0,     eta - pi,     eta + pi + 2 * pi / 3,     eta + pi - 2 * pi / 3]
        #self.betas =    [0,     pi,     0,    2 * pi / 3,         -2 * pi / 3]
        #self.thetas =   [pi,    pi / 2,     pi / 2 + 2 * pi / 3,     pi / 2 - 2 * pi / 3]
        #self.phis =     [0,     pi,     pi + 2 * pi / 3,     pi - 2 * pi / 3]
        #Viviani
        self.alphas =   [0,     0,     0,     0,     0]
        self.betas =    [pi/4,    -pi/4,     3*pi/4,    -3*pi/4,         0]
        self.thetas =   [0,     0,     0,     0]
        self.phis =     [-pi/4,    pi/4,    -3*pi/4,  3*pi/4 ]
        
        self.indices_list = []
        self.n_repetitions = 1
        self.listvert=[]
        
    def add_witness_circuits(self,listvert) -> None:
        self.listvert=listvert
        self._get_angles_lists()

        self.circuits.clear()
        #print(*self.indices_list)
        for s in range(20*self.n_repetitions):
            # self.circuits.append(QuantumCircuit(127, len(listvert)))
            self.circuits.append(QuantumCircuit(2, len(listvert)))  # TR: For tests
            cbit=0
            q=0
            # TB: Usunąłem odejmowanie poprzedniego kąta, wtedy symulator daje poprawne wyniki
            for qubit in listvert:
            
                self._s_gate(self.circuits[-1], self.alphas[self.indices_list[q][s][0]], qubit)
                self._s_gate(self.circuits[-1], self.betas[self.indices_list[q][s][0]], qubit)
                self._s_gate(self.circuits[-1], self.phis[self.indices_list[q][s][1]], qubit)
                self._s_gate(self.circuits[-1], self.thetas[self.indices_list[q][s][1]], qubit)
                self.circuits[-1].measure(qubit,cbit)
                cbit+=1
                q+=1
            
            #self.circuits[-1].reset(qubit=0)
            #self.circuits[-1].reset(qubit=0)
            #self.circuits[-1].reset(qubit=0)
            
            
    # TB: Dodałem randomizację kolejności układów w jobie
    def _get_angles_lists(self):  
        for v in self.listvert:
            self.va=[]
            for n in range(self.n_repetitions):           
                for i in range(len(self.alphas)):
                    for j in range(len(self.thetas)):
                        self.va.append([i,j])
            
            random.shuffle(self.va)
            #print(*self.va)
            self.indices_list.append(self.va)

    def _s_gate(self, circuit: QuantumCircuit, theta: float, qubit: int):
        # TB: zamiana bramki U na RZ*SX*RZ
        circuit.rz(theta, qubit)
        circuit.sx(qubit)
        circuit.rz(-theta, qubit)
    def _s_gatelast(self, circuit: QuantumCircuit, theta: float):
        # TB: zamiana bramki U na RZ*SX*RZ
        circuit.rz(theta, qubit=0)
        circuit.sx(qubit=0)
        
    def save_to_file(self, csv_path, zip_filename):
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_j =[]
        print(self.n_repetitions)
        listvert=self.listvert
        for s in range(20*self.n_repetitions):
            iva=0
            ivb=0
            for q in range(len(listvert)):
                iva+=5**q*self.indices_list[q][s][0]
                ivb+=4**q*self.indices_list[q][s][1]
            indices_i.append(iva)
            indices_j.append(ivb)
        pandas_table["i"] = indices_i
        pandas_table["j"] = indices_j
        
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
            
class WitnessJobParameterized(WitnessJob):
    def add_witness_circuits(self, parameters) -> None:
        angles_dicts = self._get_angles_lists(parameters)

        self.circuits.clear()
        
        for angles_dict in angles_dicts:
            self.circuits.append(QuantumCircuit(1, 1))
            
            self.circuits[-1].reset(qubit=0)
            
            self._s_gate(self.circuits[-1], angles_dict["alpha"])
            self._s_gate(self.circuits[-1], angles_dict["beta"])
            self.circuits[-1].barrier()
            self._s_gate(self.circuits[-1], angles_dict["phi"])
            self._s_gate(self.circuits[-1], angles_dict["theta"])
            self.circuits[-1].measure_all()
            
            self.circuits[-1].reset(qubit=0)
    
    def _get_angles_lists(self, parameters) -> List[Dict[str, float]]:

        angles_dicts = []
        
        # TB: Zapisuję również indeks n żeby móc sprawdzić jaka wartość parametru była użyta
        for k in range(self.n_repetitions):
            for n in range(len(parameters)+len(self.alphas)-1):
                for j in range(len(self.thetas)):
                    self.indices_list.append([n,j])
        self.indices_list = np.array(self.indices_list)
            
        np.random.shuffle(self.indices_list)
                
        for n, j in self.indices_list:
            angles_dicts.append({})
            
            # TB: Zamiast alphas[1] dodajemy parametr
            if n ==0:
                angles_dicts[-1]["alpha"] = self.alphas[n]
                angles_dicts[-1]["beta"] = self.betas[n]
            elif n<4:
                angles_dicts[-1]["alpha"] = self.alphas[n+1]
                angles_dicts[-1]["beta"] = self.betas[n+1]
            else:
                angles_dicts[-1]["alpha"] = parameters[n-4]
                angles_dicts[-1]["beta"] = pi/2 + parameters[n-4]                              
            angles_dicts[-1]["phi"] = self.phis[j]
            angles_dicts[-1]["theta"] = self.thetas[j]

        return angles_dicts
    
    def save_to_file(self, csv_path, zip_filename):
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        
        indices_n, indices_j = self.indices_list.transpose()
        pandas_table["n"] = indices_n
        pandas_table["j"] = indices_j
        
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
