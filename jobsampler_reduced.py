import os
import random
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy import pi
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister



class Job:
    # TODO TR: Specify the types, just as in circuits.
    parameters_list = []
    circuits: List[QuantumCircuit] = []
    last_status = None
    queued_job = None
    status = None
    test_circuits_number = None
    if_saved: bool = False

    def __init__(self) -> None:
        self.parameters_list = []
        self.circuits = []
        self.status = None
        self.queued_job = None
        self.last_status = None
        self.test_circuits_number = None
        self.if_saved = False
        #self.log_filename = log_filename

    def update_status(self):
        status_before_update = self.last_status
        self.last_status = self.queued_job.status().name

        if_changed = True
        if self.last_status == status_before_update:
            if_changed = False

        return if_changed

    # Zapis danych do pliku
    def save_to_file(self, csv_path, zip_filename):
        results = self.queued_job.result().get_counts()
        tabela = pd.DataFrame.from_dict(results).fillna(0)

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

        csv_filename = csv_path.split("/")[-1]
        with ZipFile(zip_filename + ".zip", "a") as plik_zip:
            plik_zip.write(csv_path, arcname="results/" + csv_filename)

        self.if_saved = True

        try:
            os.remove(csv_path)
        except:
            save_to_log(self.log_filename, f"Error removing {csv_path}")
class WitnessJob(Job):
    def __init__(self) -> None:
        super().__init__()

        self.indices_list = []
        self.n_repetitions = 1

    def add_witness_circuits(self) -> None:
        angles_dicts = self._get_angles_lists()

        self.circuits.clear()

        for angles_dict in angles_dicts:
            self.circuits.append(QuantumCircuit(1, 1))

            self.s_gate(self.circuits[-1], angles_dict["alpha"])
            self.s_gate(self.circuits[-1], angles_dict["beta"])
            self.circuits[-1].barrier()
            self.s_gate(self.circuits[-1], angles_dict["phi"])
            self.s_gate_last(self.circuits[-1], angles_dict["theta"])
            self.circuits[-1].measure_all()

    def _get_angles_lists(self) -> List[Dict[str, float]]:

        angles_dicts = []

        for n in range(self.n_repetitions):
            for i in range(len(self.alphas)):
                for j in range(len(self.thetas)):
                    self.indices_list.append([i, j])
        self.indices_list = np.array(self.indices_list)

        np.random.shuffle(self.indices_list)

        for i, j in self.indices_list:
            angles_dicts.append({})
            angles_dicts[-1]["alpha"] = self.alphas[i]
            angles_dicts[-1]["beta"] = self.betas[i]
            angles_dicts[-1]["phi"] = self.phis[j]
            angles_dicts[-1]["theta"] = self.thetas[j]

        return angles_dicts
    def _get_angles_lists_single(self) -> List[Dict[str, float]]:

        angles_dicts = []

        for n in range(self.n_repetitions):
            for i in range(len(self.alphas)):
                    self.indices_list.append(i)
        self.indices_list = np.array(self.indices_list)

        np.random.shuffle(self.indices_list)

        for i in self.indices_list:
            angles_dicts.append({})
            angles_dicts[-1]["alpha"] = self.alphas[i]
            angles_dicts[-1]["beta"] = self.betas[i]

        return angles_dicts

    @staticmethod
    def s_gate(circuit: QuantumCircuit, theta: float, qubit=0):
        circuit.rz(theta, qubit=qubit)
        circuit.sx(qubit=qubit)
        circuit.rz(-theta, qubit=qubit)

    @staticmethod
    def s_gate_last(circuit: QuantumCircuit, theta: float):
        circuit.rz(theta, qubit=0)
        circuit.sx(qubit=0)

    def save_to_file(self, csv_path, zip_filename):
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)

        indices_i, indices_j = self.indices_list.transpose()
        pandas_table["i"] = indices_i
        pandas_table["j"] = indices_j

        # Saving to file
        pandas_table.to_csv(csv_path)
        csv_filename = csv_path.split("/")[-1]
        with ZipFile(zip_filename + ".zip", "a") as plik_zip:
            plik_zip.write(csv_path, arcname="results/" + csv_filename)

        self.if_saved = True

        try:
            os.remove(csv_path)
        except:
            save_to_log(self.log_filename, f"Error removing {csv_path}")

class VivianiJob(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubit= 0
    
    def add_witness_circuits(self, qubit: int) -> None:
        self.qubit = qubit
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(20 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            cr=ClassicalRegister(1, "cr")
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, cr))
            
            self.s_gate(
                self.circuits[-1], self.alphas[self.indices_list[s][0]], qubit
            )
            self.s_gate(
                self.circuits[-1], self.betas[self.indices_list[s][0]], qubit
            )
            self.s_gate(
                self.circuits[-1], self.phis[self.indices_list[s][1]], qubit
            )
            self.s_gate(
                self.circuits[-1], self.thetas[self.indices_list[s][1]], qubit
            )
            self.circuits[-1].measure(qubit,cr)

    def _get_angles_lists(self):
        self.va = []
        for n in range(self.n_repetitions):
            for i in range(len(self.alphas)):
                for j in range(len(self.thetas)):
                    self.va.append([i, j])

        random.shuffle(self.va)
        # print(*self.va)
        self.indices_list=self.va

    def save_to_file(self, csv_path, zip_filename):
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i = []
        indices_j = []
        print(self.n_repetitions)
        for s in range(20 * self.n_repetitions):
            iva =  self.indices_list[s][0]
            ivb = self.indices_list[s][1]
            indices_i.append(iva)
            indices_j.append(ivb)
        pandas_table["i"] = indices_i
        pandas_table["j"] = indices_j

        # Saving to file
        pandas_table.to_csv(csv_path)
        csv_filename = csv_path.split("/")[-1]
        with ZipFile(zip_filename + ".zip", "a") as plik_zip:
            plik_zip.write(csv_path, arcname="results/" + csv_filename)

        self.if_saved = True

        try:
            os.remove(csv_path)
        except Exception as alert:
            print(alert)
class BellJob(WitnessJob,):
    def __init__(self) -> None:
        super().__init__()
        self.n_repetitions = 1
        self.qubit = 0
    def add_witness_circuits(self, qubit: int) -> None:
        #porzebujemy qubiow od qubit-3 do qubit+3, qubit jest zrodlem
        self.qubit = qubit
        self.circuits.clear()
        eta=np.arccos(1/np.sqrt(3))
        for s in range(self.n_repetitions):
            cr=ClassicalRegister(6, "cr"))
            qreg = QuantumRegister(127) #AQT pewnie 20
            self.circuits.append(QuantumCircuit(qreg, cr))

            self.circuits[-1].sx(qubit)
            self.cx(self.circuits[-1],qubit, qubit+1)
            self.cx(self.circuits[-1], qubit, qubit-1)
            self.cx(self.circuits[-1], qubit-1, qubit)
            self.circuits[-1].rz(-np.pi/2,qubit+1)
            self.circuits[-1].x(qubit+1)
            self.cx(self.circuits[-1],qubit-1,qubit-2)
            self.cx(self.circuits[-1],qubit+1,qubit+2)
            self.circuits[-1].rz(np.pi/2,qubit-1)
            self.circuits[-1].rz(np.pi/2,qubit+1)
            self.circuits[-1].sx(qubit-1)
            self.circuits[-1].sx(qubit+1)
            self.circuits[-1].rz(-np.pi/4,qubit-2)
            self.circuits[-1].rz(-np.pi/4,qubit+2)
            self.circuits[-1].sx(qubit-3)
            self.circuits[-1].sx(qubit+3)
            self.circuits[-1].rz(eta,qubit-3)
            self.circuits[-1].rz(eta,qubit+3)
            self.circuits[-1].sx(qubit-3)
            self.circuits[-1].sx(qubit+3)
            self.circuits[-1].rz(np.pi/4,qubit-3)
            self.circuits[-1].rz(np.pi/4,qubit+3)
            self.cx(self.circuits[-1],qubit-3,qubit-2)
            self.cx(self.circuits[-1],qubit+3,qubit+2)

            self.circuits[-1].rz(np.pi/2,qubit-3)
            self.circuits[-1].rz(np.pi/2,qubit+3)
            self.circuits[-1].sx(qubit-3)
            self.circuits[-1].sx(qubit+3)
            self.circuits[-1].measure([qubit-1,qubit-2,qubit-3,qubit+1,qubit+2,qubit+3],cr)
    def update_status(self) -> bool:
        status_before_update = self.last_status
        try:
            self.last_status = self.queued_job.status().name
        except:
            self.last_status = self.queued_job.status()
        
        if_changed = None
        if self.last_status == status_before_update:
            if_changed = False
        else:
            if_changed = True
        return if_changed

    def save_to_file(self, csv_path, zip_filename):
        result_counts=[]
        job_result = self.queued_job.result()
        for idx, pub_result in enumerate(job_result):
            result_counts.append(getattr(pub_result.data, "cr").get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)

        
        # Saving to file
        pandas_table.to_csv(csv_path)
        csv_filename = csv_path.split('/')[-1]
        with ZipFile(zip_filename + '.zip', 'a') as plik_zip:
            plik_zip.write(csv_path, arcname='results/' + csv_filename)
        self.if_saved = True

        try:
            os.remove(csv_path)
        except Exception as alert:
            print(alert)
