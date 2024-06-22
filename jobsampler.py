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
        self.res=0
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
                #if self.res==1:
                #self.circuits[-1].reset([qubit-3,qubit-2,qubit-1,qubit,qubit+1,qubit+2,qubit+3,qubit+4])
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
            for i in range(len(self.qubits_list)):
                result_counts.append(getattr(pub_result.data, "cr"+str(i)).get_counts())
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_q=[]
        #qubits_list=self.qubits_list
        for s in range(self.n_repetitions):
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
                self.circuits[-1].reset([qubit,qubit+1])
                self.circuits[-1].reset([qubit,qubit+1])
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
                self.circuits[-1].reset([qubit,qubit+1])
                self.circuits[-1].reset([qubit,qubit+1])
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
