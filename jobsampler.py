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

    def add_test_circuits(self, test_number: int) -> None:
        self.test_circuits_number = test_number

        circuit_test_0 = QuantumCircuit(1, 1)
        circuit_test_0.measure(0, 0)

        circuit_test_1 = QuantumCircuit(1, 1)
        circuit_test_1.x(0)
        circuit_test_1.measure(0, 0)

        for j in range(0, test_number):
            self.circuits.append(circuit_test_0)
            self.circuits.append(circuit_test_1)

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


class RotationJob(Job):

    def __init__(self):
        Job.__init__(self)
        self.theta_range = np.linspace(-np.pi, 7 * np.pi / 8, 16)

    def add_rotation_circuits(self, parameters_list: List[float]) -> None:
        self.parameters_list = parameters_list

        theta = Parameter("t")
        circuit_rotation = QuantumCircuit(1, 1)
        circuit_rotation.sx(0)
        circuit_rotation.rz(theta + np.pi, 0)
        circuit_rotation.sx(0)
        circuit_rotation.measure(0, 0)

        for t in self.parameters_list:
            # self.circuits.append(circuit_rotation.bind_parameters({theta: t}))
            # TODO TR: Test it!
            theta.bind({"t": t})
            self.circuits.append(circuit_rotation)

    def add_test_circuits_with_rotations(self) -> None:
        """
        Prepares test circuits, as described in 22 VI e-mail, by prof. AB. In each
        circuit we start with |0> and then apply rz(theta) and X gate n number of
        times when n = 0, ..., 100. We expect to run everything on Lima in one job.
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
    def __init__(self) -> None:
        super().__init__()

        # self.alphas =   [0,     2 * pi / 3, 2 * pi / 3, -2 * pi / 3,    -2 * pi / 3]
        # self.betas =    [0,     pi / 6,     -pi / 6,    pi / 6,         -pi / 6]
        # self.thetas =   [2 * pi / 3 + pi,    2 * pi / 3 + pi,  -2 * pi / 3 + pi,   -2 * pi / 3 + pi]
        # self.phis =     [pi / 6 + pi,        -pi / 6 + pi,     pi / 6 + pi,       -pi / 6 + pi]

        # Optimized angles
        eta = 1.23095942
        # self.alphas =   [0,     0,     eta - pi,     eta + pi + 2 * pi / 3,     eta + pi - 2 * pi / 3]
        # self.betas =    [0,     pi,     0,    2 * pi / 3,         -2 * pi / 3]
        # self.thetas =   [pi,    pi / 2,     pi / 2 + 2 * pi / 3,     pi / 2 - 2 * pi / 3]
        # self.phis =     [0,     pi,     pi + 2 * pi / 3,     pi - 2 * pi / 3]
        # Viviani
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

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


class WitnessJobParameterized(WitnessJob):
    def add_witness_circuits(self, parameters) -> None:
        angles_dicts = self._get_angles_lists(parameters)

        self.circuits.clear()

        for angles_dict in angles_dicts:
            self.circuits.append(QuantumCircuit(1, 1))

            self.circuits[-1].reset(qubit=0)

            self.s_gate(self.circuits[-1], angles_dict["alpha"])
            self.s_gate(self.circuits[-1], angles_dict["beta"])
            self.circuits[-1].barrier()
            self.s_gate(self.circuits[-1], angles_dict["phi"])
            self.s_gate(self.circuits[-1], angles_dict["theta"])
            self.circuits[-1].measure_all()

            self.circuits[-1].reset(qubit=0)

    def _get_angles_lists(self, parameters) -> List[Dict[str, float]]:

        angles_dicts = []

        for k in range(self.n_repetitions):
            for n in range(len(parameters) + len(self.alphas) - 1):
                for j in range(len(self.thetas)):
                    self.indices_list.append([n, j])
        self.indices_list = np.array(self.indices_list)

        np.random.shuffle(self.indices_list)

        for n, j in self.indices_list:
            angles_dicts.append({})

            if n == 0:
                angles_dicts[-1]["alpha"] = self.alphas[n]
                angles_dicts[-1]["beta"] = self.betas[n]
            elif n < 4:
                angles_dicts[-1]["alpha"] = self.alphas[n + 1]
                angles_dicts[-1]["beta"] = self.betas[n + 1]
            else:
                angles_dicts[-1]["alpha"] = parameters[n - 4]
                angles_dicts[-1]["beta"] = pi / 2 + parameters[n - 4]
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
        self.qubits_list = []
    
    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(20 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.s_gate(
                    self.circuits[-1], self.alphas[self.indices_list[i][s][0]], qubit
                )
                self.s_gate(
                    self.circuits[-1], self.betas[self.indices_list[i][s][0]], qubit
                )
                self.s_gate(
                    self.circuits[-1], self.phis[self.indices_list[i][s][1]], qubit
                )
                self.s_gate(
                    self.circuits[-1], self.thetas[self.indices_list[i][s][1]], qubit
                )
                self.circuits[-1].measure(qubit,cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(len(self.alphas)):
                    for j in range(len(self.thetas)):
                        self.va.append([i, j])

            random.shuffle(self.va)
            # print(*self.va)
            self.indices_list.append(self.va)

    def save_to_file(self, csv_path, zip_filename):
        result_counts = self.queued_job.result().get_counts()
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i = []
        indices_j = []
        print(self.n_repetitions)
        listvert = self.qubits_list
        for s in range(20 * self.n_repetitions):
            iva = 0
            ivb = 0
            for q in range(len(listvert)):
                iva += 5**q * self.indices_list[q][s][0]
                ivb += 4**q * self.indices_list[q][s][1]
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
class VivianiJobSwap(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_lista = []
        self.qubits_listb = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_lista: List[int], qubits_listb: List[int], qubits_dir: List[int]) -> None:
        self.qubits_lista = qubits_lista
        self.qubits_listb = qubits_listb
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(20 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_lista)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_lista)):
                qubita=qubits_lista[i]
                qubitb=qubits_lista[i]
                di=qubits_dir[i]
                self.s_gate(
                    self.circuits[-1], self.alphas[self.indices_list[i][s][0]], qubita
                )
                self.s_gate(
                        self.circuits[-1], self.betas[self.indices_list[i][s][0]], qubita
                    )
                #if di>0:
                    #print(qubita,qubitb,self.circuits[-1])
                #    self.cx1(self.circuits[-1], qubita, qubitb)
                #    self.cx0(self.circuits[-1], qubitb, qubita)
                #else:
                #    self.cx0(self.circuits[-1],qubita,qubitb)
                #    self.cx1(self.circuits[-1],qubitb,qubita)
                self.s_gate(
                        self.circuits[-1], self.phis[self.indices_list[i][s][1]], qubitb
                    )
                self.s_gate(
                    self.circuits[-1], self.thetas[self.indices_list[i][s][1]], qubitb
                )
                self.circuits[-1].measure(qubitb,cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_lista:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(len(self.alphas)):
                    for j in range(len(self.thetas)):
                        self.va.append([i, j])

            random.shuffle(self.va)
            # print(*self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_lista)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_j=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(20*self.n_repetitions):
            for q in range(len(self.qubits_lista)):
                iva=self.indices_list[q][s][0]
                ivb=self.indices_list[q][s][1]
                indices_i.append(iva)
                indices_j.append(ivb)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["j"] = indices_j
        pandas_table["q"] = indices_q
        
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
class CayleyJob:
    def __init__(self) -> None:
        self.n_repetitions = 1
        self.indices_list = []
        self.qubits_list=[]
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
        for v in self.qubits_list:
            self.va=[]
            for n in range(self.n_repetitions):           
                for i in range(11):
                    self.va.append(i)
            random.shuffle(self.va)
            #print(*self.va)
            self.indices_list.append(self.va)

    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list=qubits_list
        self._generate_indices_list()
        #print(len(self.indices_list))
        self.circuits.clear()
        #qubit = 0
        #circuit = QuantumCircuit(127, 10)
        for s in range(11*self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(qreg,meas))
            self.circuits.append(QuantumCircuit(qreg, *cr))
            #self.circuits.append(QuantumCircuit(127, len(qubits_list)))
            cbit=0
            q=0
            for qubit in qubits_list:
                #qubit=qreg[qub]
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
                self.circuits[-1].measure(qubit,cr[cbit])
                cbit+=1
                q+=1
            #self.circuits[-1].measure([qreg[i] for i in qubits_list],meas)
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
        
    def save_to_file(self, csv_path: str, zip_filename: str) -> None:     
        #result_counts = self.queued_job.result().get_counts()
        result_counts=[]
        #print(self.queued_job.result())
        job_result = self.queued_job.result()
        for idx, pub_result in enumerate(job_result):
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        qubits_list=self.qubits_list
        for s in range(11*self.n_repetitions):
            for q in range(len(qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellJob(WitnessJob,):
    def __init__(self) -> None:
        super().__init__()
        self.n_repetitions = 1
        self.qubits_list = []
        self.res=1
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int], qubits_dir: List[List[int]],res:int) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self.res=res
        self.circuits.clear()
        eta=np.arccos(1/np.sqrt(3))
        for s in range(self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(6, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                if s%3<res-1:
                    self.circuits[-1].reset([qubit-3,qubit-2,qubit-1,qubit,qubit+1,qubit+2,qubit+3])
                    continue
                di=qubits_dir[i]
                self.circuits[-1].sx(qubit)
                if di[3]:
                    self.cx1(self.circuits[-1],qubit, qubit+1)
                else:
                    self.cx0(self.circuits[-1],qubit, qubit+1)
                if di[2]:
                    self.cx0(self.circuits[-1], qubit, qubit-1)
                    self.cx1(self.circuits[-1], qubit-1, qubit)
                else:
                    self.cx1(self.circuits[-1],qubit,qubit-1)
                    self.cx0(self.circuits[-1],qubit-1,qubit)
                self.circuits[-1].rz(-np.pi/2,qubit+1)
                self.circuits[-1].x(qubit+1)
                if di[1]:
                    self.cx0(self.circuits[-1],qubit-1,qubit-2)
                else:
                    self.cx1(self.circuits[-1],qubit-1,qubit-2)
                if di[4]:
                    self.cx1(self.circuits[-1],qubit+1,qubit+2)
                else:
                    self.cx0(self.circuits[-1],qubit+1,qubit+2)
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
                if di[0]:
                    self.cx1(self.circuits[-1],qubit-3,qubit-2)
                else:
                    self.cx0(self.circuits[-1],qubit-3,qubit-2)
                if di[5]:
                    self.cx0(self.circuits[-1],qubit+3,qubit+2)
                else:
                    self.cx1(self.circuits[-1],qubit+3,qubit+2)
                self.circuits[-1].rz(np.pi/2,qubit-3)
                self.circuits[-1].rz(np.pi/2,qubit+3)
                self.circuits[-1].sx(qubit-3)
                self.circuits[-1].sx(qubit+3)
                self.circuits[-1].measure([qubit-1,qubit-2,qubit-3,qubit+1,qubit+2,qubit+3],cr[i])
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
            if idx%3<self.res-1:
                continue
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(self.n_repetitions//self.res):
            for q in range(len(self.qubits_list)):
                indices_q.append(q)
        pandas_table["q"] = indices_q
        
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
class BellJobD(WitnessJob,):
    def __init__(self) -> None:
        super().__init__()
        self.n_repetitions = 1
        self.qubits_list = []
        self.res=1
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int], qubits_dir: List[List[int]],res:int) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self.res=res
        self.circuits.clear()
        eta=np.arccos(1/np.sqrt(3))
        for s in range(self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(6, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                if s%3<res-1:
                    self.circuits[-1].reset([qubit-3,qubit-2,qubit-1,qubit,qubit+1,qubit+2,qubit+3,qubit+4])
                    continue
                di=qubits_dir[i]
                self.circuits[-1].sx(qubit)
                if di[3]:
                    self.cx1(self.circuits[-1],qubit, qubit+1)
                else:
                    self.cx0(self.circuits[-1],qubit, qubit+1)
                if di[2]:
                    self.cx0(self.circuits[-1], qubit, qubit-1)
                    self.cx1(self.circuits[-1], qubit-1, qubit)
                else:
                    self.cx1(self.circuits[-1],qubit,qubit-1)
                    self.cx0(self.circuits[-1],qubit-1,qubit)
                if di[4]:
                    self.cx1(self.circuits[-1],qubit+1, qubit+2)
                    self.cx0(self.circuits[-1],qubit+2, qubit+1)
                else:
                    self.cx0(self.circuits[-1],qubit+1, qubit+2)
                    self.cx1(self.circuits[-1],qubit+2, qubit+1)
                self.circuits[-1].rz(-np.pi/2,qubit+2)
                self.circuits[-1].x(qubit+2)
                if di[1]:
                    self.cx0(self.circuits[-1],qubit-1,qubit-2)
                else:
                    self.cx1(self.circuits[-1],qubit-1,qubit-2)
                if di[5]:
                    self.cx1(self.circuits[-1],qubit+2,qubit+3)
                else:
                    self.cx0(self.circuits[-1],qubit+2,qubit+3)
                self.circuits[-1].rz(np.pi/2,qubit-1)
                self.circuits[-1].rz(np.pi/2,qubit+2)
                self.circuits[-1].sx(qubit-1)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].rz(-np.pi/4,qubit-2)
                self.circuits[-1].rz(-np.pi/4,qubit+3)
                self.circuits[-1].sx(qubit-3)
                self.circuits[-1].sx(qubit+4)
                self.circuits[-1].rz(eta,qubit-3)
                self.circuits[-1].rz(eta,qubit+4)
                self.circuits[-1].sx(qubit-3)
                self.circuits[-1].sx(qubit+4)
                self.circuits[-1].rz(np.pi/4,qubit-3)
                self.circuits[-1].rz(np.pi/4,qubit+4)
                if di[0]:
                    self.cx1(self.circuits[-1],qubit-3,qubit-2)
                else:
                    self.cx0(self.circuits[-1],qubit-3,qubit-2)
                if di[6]:
                    self.cx0(self.circuits[-1],qubit+4,qubit+3)
                else:
                    self.cx1(self.circuits[-1],qubit+4,qubit+3)
                self.circuits[-1].rz(np.pi/2,qubit-3)
                self.circuits[-1].rz(np.pi/2,qubit+4)
                self.circuits[-1].sx(qubit-3)
                self.circuits[-1].sx(qubit+4)
                self.circuits[-1].measure([qubit-1,qubit-2,qubit-3,qubit+2,qubit+3,qubit+4],cr[i])
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
            if idx%3<self.res-1:
                continue
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(self.n_repetitions//self.res):
            for q in range(len(self.qubits_list)):
                indices_q.append(q)
        pandas_table["q"] = indices_q
        
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
class Single(WitnessJob):
    def __init__(self) -> None:
        super().__init__()
        eta=np.arccos(1/np.sqrt(3))
        # Viviani optimized angles
        self.alphas = [0, eta, -eta, eta+np.pi/2,np.pi/2-eta]
        self.betas = [0,np.pi,np.pi,np.pi/2,np.pi/2]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int], qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()
        eta=np.arccos(1/np.sqrt(3))
        self.circuits.clear()
        for s in range(5 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                di=qubits_dir[i]
                self.s_gate(
                    self.circuits[-1], self.alphas[self.indices_list[i][s]], qubit
                )
                self.s_gate(
                        self.circuits[-1], self.betas[self.indices_list[i][s]], qubit
                    )
                #if di[0]>0:
                #    #print(qubita,qubitb,self.circuits[-1])
                #    self.cx1(self.circuits[-1], qubit, qubit+1)
                #    self.cx0(self.circuits[-1], qubit+1, qubit)
                #else:
                #    self.cx0(self.circuits[-1],qubit,qubit+1)
                #    self.cx1(self.circuits[-1],qubit+1,qubit)
                if di[0]:
                    self.cx1(self.circuits[-1],qubit,qubit+1)
                else:
                    self.cx0(self.circuits[-1],qubit,qubit+1)
                self.circuits[-1].rz(np.pi/2,qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].rz(-np.pi/4,qubit+1)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].rz(eta,qubit+2)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].rz(np.pi/4,qubit+2)
                if di[1]:
                    self.cx0(self.circuits[-1],qubit+2,qubit+1)
                else:
                    self.cx1(self.circuits[-1],qubit+2,qubit+1)
                self.circuits[-1].rz(np.pi/2,qubit+2)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].measure([qubit,qubit+1,qubit+2],cr[i])


    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(len(self.alphas)):
                        self.va.append(i)

            random.shuffle(self.va)
            # print(*self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(5*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class SingleS(WitnessJob):
    def __init__(self) -> None:
        super().__init__()
        eta=np.arccos(1/np.sqrt(3))
        # Viviani optimized angles
        self.alphas = [0, eta, -eta, eta+np.pi/2,np.pi/2-eta]
        self.betas = [0,np.pi,np.pi,np.pi/2,np.pi/2]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int], qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()
        eta=np.arccos(1/np.sqrt(3))
        self.circuits.clear()
        for s in range(5 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                di=qubits_dir[i]
                self.s_gate(
                    self.circuits[-1], self.alphas[self.indices_list[i][s]], qubit
                )
                self.s_gate(
                        self.circuits[-1], self.betas[self.indices_list[i][s]], qubit
                    )
                if di[0]>0:
                    #print(qubita,qubitb,self.circuits[-1])
                    self.cx1(self.circuits[-1], qubit, qubit+1)
                    self.cx0(self.circuits[-1], qubit+1, qubit)
                else:
                    self.cx0(self.circuits[-1],qubit,qubit+1)
                    self.cx1(self.circuits[-1],qubit+1,qubit)
                if di[1]:
                    self.cx1(self.circuits[-1],qubit+1,qubit+2)
                else:
                    self.cx0(self.circuits[-1],qubit+1,qubit+2)
                self.circuits[-1].rz(np.pi/2,qubit+1)
                self.circuits[-1].sx(qubit+1)
                self.circuits[-1].rz(-np.pi/4,qubit+2)
                self.circuits[-1].sx(qubit+3)
                self.circuits[-1].rz(eta,qubit+3)
                self.circuits[-1].sx(qubit+3)
                self.circuits[-1].rz(np.pi/4,qubit+3)
                if di[2]:
                    self.cx0(self.circuits[-1],qubit+3,qubit+2)
                else:
                    self.cx1(self.circuits[-1],qubit+3,qubit+2)
                self.circuits[-1].rz(np.pi/2,qubit+3)
                self.circuits[-1].sx(qubit+3)
                self.circuits[-1].measure([qubit+1,qubit+2,qubit+3],cr[i])


    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(len(self.alphas)):
                        self.va.append(i)

            random.shuffle(self.va)
            # print(*self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(5*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellSig(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                #self.circuits[-1].sx(qubit)
                #if di[1]:
                #    self.cx1(self.circuits[-1],qubit,qubit+1)
                #else:
                #    self.cx0(self.circuits[-1],qubit,qubit+1)
                #if di[0]:
                #    self.cx0(self.circuits[-1],qubit,qubit-1)
                #    self.cx1(self.circuits[-1],qubit-1,qubit)
                #else:
                #    self.cx1(self.circuits[-1],qubit,qubit-1)
                #    self.cx0(self.circuits[-1],qubit-1,qubit)
                #if di[2]:
                #    self.cx1(self.circuits[-1],qubit+1,qubit+2)
                #    self.cx0(self.circuits[-1],qubit+2,qubit+1)
                #else:
                #    self.cx0(self.circuits[-1],qubit+1,qubit+2)
                #    self.cx1(self.circuits[-1],qubit+2,qubit+1)
                self.circuits[-1].rz(alpha, qubit-1)
                self.circuits[-1].rz(beta, qubit+1)
                self.circuits[-1].sx(qubit-1)
                self.circuits[-1].sx(qubit+1)
                self.circuits[-1].measure([qubit-1,qubit+1],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellSigS(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                #self.circuits[-1].sx(q[1])
                #if di[1]:
                #    self.cx1(self.circuits[-1],q[1],q[2])
                #else:
                #    self.cx0(self.circuits[-1],q[1],q[2])
                #if di[0]:
                #    self.cx0(self.circuits[-1],q[1],q[0])
                #    self.cx1(self.circuits[-1],q[0],q[1])
                #else:
                #    self.cx1(self.circuits[-1],q[1],q[0])
                #    self.cx0(self.circuits[-1],q[0],q[1])
                #if di[2]:
                #    self.cx1(self.circuits[-1],q[2],q[3])
                #    self.cx0(self.circuits[-1],q[3],q[2])
                #else:
                #    self.cx0(self.circuits[-1],q[2],q[3])
                #    self.cx1(self.circuits[-1],q[3],q[2])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[2])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].measure([q[0],q[2]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellSigS6(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(-3*np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        #c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)
        


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi
                #beta=b*np.pi/2
                beta=(2*b-1)*np.pi/2
                self.circuits[-1].sx(q[3])
                if di[3]:
                    self.cx1(self.circuits[-1],q[3],q[4])
                else:
                    self.cx0(self.circuits[-1],q[3],q[4])
                if di[2]:
                    self.sw0(self.circuits[-1],q[3],q[2])
                else:
                    self.sw1(self.circuits[-1],q[3],q[2])
                if di[1]:
                    self.sw0(self.circuits[-1],q[2],q[1])
                else:
                    self.sw1(self.circuits[-1],q[2],q[1])
                if di[0]:
                    self.sw0(self.circuits[-1],q[1],q[0])
                else:
                    self.sw1(self.circuits[-1],q[1],q[0])
                if di[4]:
                    self.sw1(self.circuits[-1],q[4],q[5])
                else:
                    self.sw0(self.circuits[-1],q[4],q[5])
                if di[5]:
                    self.sw1(self.circuits[-1],q[5],q[6])
                else:
                    self.sw0(self.circuits[-1],q[5],q[6])
                if di[6]:
                    self.sw1(self.circuits[-1],q[6],q[7])
                else:
                    self.sw0(self.circuits[-1],q[6],q[7])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[7])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[7])
                self.circuits[-1].measure([q[0],q[7]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellSigSM(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(-3*np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        #c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)
        


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                #beta=b*np.pi/2
                beta=(2*b-1)*np.pi/4
                nq=len(q)//2-1
                self.circuits[-1].sx(q[nq])
                if di[nq]:
                    self.cx1(self.circuits[-1],q[nq],q[nq+1])
                else:
                    self.cx0(self.circuits[-1],q[nq],q[nq+1])
                for z in reversed(range(nq)):
                    if di[z]:
                        self.sw0(self.circuits[-1],q[z+1],q[z])
                    else:
                        self.sw1(self.circuits[-1],q[z+1],q[z])
                for z in range(nq+1,2*nq+1):  
                    if di[z]:
                        self.sw1(self.circuits[-1],q[z],q[z+1])
                    else:
                        self.sw0(self.circuits[-1],q[z],q[z+1])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[-1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[-1])
                self.circuits[-1].measure([q[0],q[-1]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellSigS2(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(-3*np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)
        


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=b*np.pi/2
                #beta=(2*b-1)*np.pi/4
                self.circuits[-1].sx(q[2])
                if di[2]:
                    self.cx1(self.circuits[-1],q[2],q[3])
                else:
                    self.cx0(self.circuits[-1],q[2],q[3])
                if di[1]:
                    self.sw0(self.circuits[-1],q[2],q[1])
                else:
                    self.sw1(self.circuits[-1],q[2],q[1])
                if di[0]:
                    self.sw0(self.circuits[-1],q[1],q[0])
                else:
                    self.sw1(self.circuits[-1],q[1],q[0])
                if di[3]:
                    self.sw1(self.circuits[-1],q[3],q[4])
                else:
                    self.sw0(self.circuits[-1],q[3],q[4])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[4])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].measure([q[0],q[4]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class SigS2(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(-3*np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)

    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                #self.circuits[-1].sx(q[0])
                #self.circuits[-1].sx(q[1])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].measure([q[0],q[1]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellObj(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(6, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                self.circuits[-1].sx(q[3])
                if di[3]:
                    self.cx1(self.circuits[-1],q[3],q[4])
                else:
                    self.cx0(self.circuits[-1],q[3],q[4])
                if di[2]:
                    self.cx1(self.circuits[-1],q[3],q[2])
                    self.cx0(self.circuits[-1],q[2],q[3])
                else:
                    self.cx0(self.circuits[-1],q[3],q[2])
                    self.cx1(self.circuits[-1],q[2],q[3])
                self.circuits[-1].rz(alpha, q[2])
                self.circuits[-1].rz(beta, q[4])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].rz(-alpha, q[2])
                self.circuits[-1].rz(-beta, q[4])
                if di[0]:
                    self.cx1(self.circuits[-1],q[2],q[0])
                else:
                    self.cx0(self.circuits[-1],q[2],q[0])
                if di[1]:
                    self.cx1(self.circuits[-1],q[2],q[1])
                else:
                    self.cx0(self.circuits[-1],q[2],q[1])
                if di[4]:
                    self.cx1(self.circuits[-1],q[4],q[5])
                else:
                    self.cx0(self.circuits[-1],q[4],q[5])
                if di[5]:
                    self.cx1(self.circuits[-1],q[4],q[6])
                else:
                    self.cx0(self.circuits[-1],q[4],q[6])
                self.circuits[-1].measure([q[0],q[1],q[2],q[5],q[6],q[4]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class BellObjM(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles
        self.alphas = [0, 0, 0, 0, 0]
        self.betas = [pi / 4, -pi / 4, 3 * pi / 4, -3 * pi / 4, 0]
        self.thetas = [0, 0, 0, 0]
        self.phis = [-pi / 4, pi / 4, -3 * pi / 4, 3 * pi / 4]

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
        self.friends = []
        self.friends_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        #c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int], friends: List[int],  friends_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self.friends = friends
        self.friends_dir = friends_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(6, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                f=friends[i]
                fi=friends_dir[i]
                if 2 in di:
                    print("2222")
                    return
                if 2 in fi:
                    print("2222")
                    return
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi
                beta=(2*b-1)*np.pi/2
                nq=len(q)//2
                self.circuits[-1].sx(q[nq])
                if di[nq]:
                    self.cx1(self.circuits[-1],q[nq],q[nq+1])
                else:
                    self.cx0(self.circuits[-1],q[nq],q[nq+1])
                for z in reversed(range(nq)):
                    if di[z]:
                        self.sw0(self.circuits[-1],q[z+1],q[z])
                    else:
                        self.sw1(self.circuits[-1],q[z+1],q[z])
                for z in range(nq+1,2*nq):  
                    if di[z]:
                        self.sw1(self.circuits[-1],q[z],q[z+1])
                    else:
                        self.sw0(self.circuits[-1],q[z],q[z+1])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[-1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[-1])
                self.circuits[-1].rz(-alpha, q[0])
                self.circuits[-1].rz(-beta, q[-1])
                if fi[0]:
                    self.cx1(self.circuits[-1],q[0],f[0])
                else:
                    self.cx0(self.circuits[-1],q[0],f[0])
                if fi[1]:
                    self.cx1(self.circuits[-1],q[0],f[1])
                else:
                    self.cx0(self.circuits[-1],q[0],f[1])
                if fi[2]:
                    self.cx1(self.circuits[-1],q[-1],f[2])
                else:
                    self.cx0(self.circuits[-1],q[-1],f[2])
                if fi[3]:
                    self.cx1(self.circuits[-1],q[-1],f[3])
                else:
                    self.cx0(self.circuits[-1],q[-1],f[3])
                self.circuits[-1].measure([f[0],f[1],q[0],f[2],f[3],q[-1]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(4*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class I3322(WitnessJob):
    def __init__(self) -> None:
        super().__init__()
        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        #c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(9 * self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                if 2 in di:
                    print("2222")
                    return
                par=self.indices_list[i][s]
                a=par%3
                b=par//3
                alpha=2*a*np.pi/3
                beta=(3-4*b)*np.pi/6
                nq=len(q)//2
                self.circuits[-1].sx(q[nq])
                if di[nq]:
                    self.cx1(self.circuits[-1],q[nq],q[nq+1])
                else:
                    self.cx0(self.circuits[-1],q[nq],q[nq+1])
                for z in reversed(range(nq)):
                    if di[z]:
                        self.sw0(self.circuits[-1],q[z+1],q[z])
                    else:
                        self.sw1(self.circuits[-1],q[z+1],q[z])
                for z in range(nq+1,len(q)-1):  
                    if di[z]:
                        self.sw1(self.circuits[-1],q[z],q[z+1])
                    else:
                        self.sw0(self.circuits[-1],q[z],q[z+1])
                self.circuits[-1].rz(alpha, q[0])
                self.circuits[-1].rz(beta, q[-1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].sx(q[-1])

                self.circuits[-1].measure([q[0],q[-1]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(9):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(9*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class tele(WitnessJob):
    def __init__(self) -> None:
        super().__init__()
        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.x(i)
        c.sx(j)
        c.ecr(i,j)
        c.rz(np.pi/2,i)
    @staticmethod
    def cx1(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,i)
        c.rz(-np.pi/2,j)
        c.sx(i)
        c.rz(np.pi/2,i)
        c.sx(i)
        c.sx(j)
        c.ecr(j,i)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
        c.sx(i)
        c.sx(j)
        c.rz(np.pi/2,i)
        c.rz(np.pi/2,j)
    def sw0(self,c: QuantumCircuit,i,j):
        #c.rz(-np.pi,j)
        self.cx0(c,i,j)
        self.cx1(c,j,i)
    def sw1(self,c: QuantumCircuit,i,j):
        self.cx1(c,i,j)
        self.cx0(c,j,i)


    def add_witness_circuits(self, qubits_list: List[int],  qubits_dir: List[int]) -> None:
        self.qubits_list = qubits_list
        self.qubits_dir = qubits_dir
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(3 * self.n_repetitions):
            cr=[]
            cra=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr"+str(i)))
            qreg = QuantumRegister(127)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                di=qubits_dir[i]
                if 2 in di:
                    print("2222")
                    return
                par=self.indices_list[i][s]
                if par==0:
                    self.circuits[-1].sx(q[0])
                elif par==1:
                    self.circuits[-1].rz(np.pi/2, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi/2, q[0])
                self.circuits[-1].rz(np.pi/2, q[1])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].rz(np.pi/2, q[1])
                if di[1]:
                    self.cx1(self.circuits[-1],q[1],q[2])
                else:
                    self.cx0(self.circuits[-1],q[1],q[2])
                if di[0]:
                    self.cx1(self.circuits[-1],q[0],q[1])
                else:
                    self.cx0(self.circuits[-1],q[0],q[1])
                self.circuits[-1].rz(np.pi/2, q[0])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].rz(np.pi/2, q[0])
                self.circuits[-1].measure([q[0],q[1]],cr[i][:2])
                with  self.circuits[-1].if_test((cr[i][1],1)):
                    self.circuits[-1].x(q[2])
                    
                with  self.circuits[-1].if_test((cr[i][0],1)):
                    self.circuits[-1].z(q[2])
                if par==0:
                    self.circuits[-1].z(q[2])
                    self.circuits[-1].sx(q[2])
                    self.circuits[-1].z(q[2])
                elif par==1:
                    self.circuits[-1].z(q[2])
                    self.circuits[-1].rz(np.pi/2, q[2])
                    self.circuits[-1].sx(q[2])
                    self.circuits[-1].rz(-np.pi/2, q[2])
                    self.circuits[-1].z(q[2])
                self.circuits[-1].measure(q[2],cr[i][2])
                

                

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(3):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(3*self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class Memory(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    
    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()
        for s in range(self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.circuits[-1].sx(qubit)
                self.circuits[-1].rz(np.pi,qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].measure(qubit,cr[i])
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.circuits[-1].measure(qubit,cr[i])
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]

                self.circuits[-1].sx(qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].measure(qubit,cr[i])
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.circuits[-1].measure(qubit,cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(4):
                        self.va.append(i)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        #print(self.indices_list)
        for s in range(4*self.n_repetitions):
            #print(s)
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
class Cross(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        # Viviani optimized angles

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    
    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()
        for s in range(self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.circuits[-1].sx(qubit)
                self.circuits[-1].rz(np.pi,qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].measure(qubit+2,cr[i])
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(1, "cr"+str(i)))
            qreg = QuantumRegister(127)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                qubit=qubits_list[i]
                self.circuits[-1].sx(qubit)
                self.circuits[-1].sx(qubit)
                self.circuits[-1].sx(qubit+2)
                self.circuits[-1].measure(qubit+2,cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(2):
                        self.va.append(i)
            self.indices_list.append(self.va)
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i=[]
        indices_q=[]
        #qubits_list=self.qubits_list
        #print(self.indices_list)
        for s in range(2*self.n_repetitions):
            #print(s)
            for q in range(len(self.qubits_list)):
                iva=self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q
        
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
