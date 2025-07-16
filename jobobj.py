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


    def update_status(self):
        status_before_update = self.last_status
        self.last_status = self.queued_job.status().name

        if_changed = True
        if self.last_status == status_before_update:
            if_changed = False

        return if_changed

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

class BellObj(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,j)
        c.sx(j)
        c.cz(i,j)
        c.z(j)
        c.sx(j)
        c.rz(np.pi/2,j)



    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(6, "cr"+str(i)))
            qreg = QuantumRegister(133)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                self.circuits[-1].sx(q[3])
                self.cx0(self.circuits[-1],q[3],q[4])              
                self.cx0(self.circuits[-1],q[3],q[2])
                self.cx0(self.circuits[-1],q[2],q[3])
                self.circuits[-1].rz(alpha, q[2])
                self.circuits[-1].rz(beta, q[4])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].rz(-alpha, q[2])
                self.circuits[-1].rz(-beta, q[4])
                self.cx0(self.circuits[-1],q[2],q[0])
                self.cx0(self.circuits[-1],q[2],q[1])
                self.cx0(self.circuits[-1],q[4],q[5])
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
class BellSig(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,j)
        c.sx(j)
        c.cz(i,j)
        c.z(j)
        c.sx(j)
        c.rz(np.pi/2,j)



    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(2, "cr"+str(i)))
            qreg = QuantumRegister(156)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                m=len(q)//2
                self.circuits[-1].sx(q[m])
                self.cx0(self.circuits[-1],q[m],q[m+1])
                for j in reversed(range(m)):
                    self.cx0(self.circuits[-1],q[j+1],q[j])
                    self.cx0(self.circuits[-1],q[j],q[j+1])
                for j in range(m+2,len(q)):
                    self.cx0(self.circuits[-1],q[j-1],q[j])
                    self.cx0(self.circuits[-1],q[j],q[j-1]) 
                
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
class tele(WitnessJob):
    def __init__(self) -> None:
        super().__init__()
        self.nnq=0
        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.dirs=[]
    @staticmethod
    def cx0(c: QuantumCircuit,i,j,d):
        c.cx(i,j)
        return
        if d==2:
            c.rz(np.pi/2,j)
            c.sx(j)
            c.cz(i,j)
            c.rz(np.pi,j)
            c.sx(j)
            c.rz(np.pi/2,j)
        elif d==0:
            c.x(i)
            c.sx(j)
            c.ecr(i,j)
            c.rz(np.pi/2,i)
        elif d==1:
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
        else:
            print(i,j,d,"WRONG!")
            c.sx(-1000)         
            return



    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()
        nnq=self.nnq
        self.circuits.clear()
        for s in range(3 * self.n_repetitions):
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr"+str(i)))
            qreg = QuantumRegister(nnq)
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                j=len(q)//2
                par=self.indices_list[i][s]
                if par:
                    self.circuits[-1].sx(q[0])
                if par==1:
                    self.circuits[-1].rz(np.pi/2, q[0])
                for k in range(j):
                    self.circuits[-1].sx(q[2*k+1])
                    self.circuits[-1].rz(np.pi/2,q[2*k+1])
                    self.cx0(self.circuits[-1],q[2*k+1],q[2*k+2],self.dirs[q[2*k+1]][q[2*k+2]])
                    self.cx0(self.circuits[-1],q[2*k],q[2*k+1],self.dirs[q[2*k]][q[2*k+1]])
                    self.circuits[-1].rz(np.pi/2,q[2*k])
                    self.circuits[-1].sx(q[2*k])
                    self.circuits[-1].measure([q[2*k],q[2*k+1]],[cr[i][0],cr[i][1]])
                    with  self.circuits[-1].if_test((cr[i][1],1)):
                        self.circuits[-1].x(q[2*k+2])
                                
                    with  self.circuits[-1].if_test((cr[i][0],1)):
                        self.circuits[-1].rz(np.pi,q[2*k+2])
                self.circuits[-1].rz(np.pi,q[2*j])
                if par==1:
                    self.circuits[-1].rz(-np.pi/2, q[2*j])
                if par:
                    self.circuits[-1].sx(q[2*j])
                self.circuits[-1].measure(q[2*j],cr[i][2])

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
class BellObj9(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,j)
        c.sx(j)
        c.cz(i,j)
        c.z(j)
        c.sx(j)
        c.rz(np.pi/2,j)



    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(4 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(18, "cr"+str(i)))
            qreg = QuantumRegister(156)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                par=self.indices_list[i][s]
                a=par%2
                b=par//2
                alpha=a*np.pi/2
                beta=(2*b-1)*np.pi/4
                self.circuits[-1].sx(q[9])
                self.cx0(self.circuits[-1],q[9],q[10])              
                self.cx0(self.circuits[-1],q[9],q[8])
                self.cx0(self.circuits[-1],q[8],q[9])
                self.circuits[-1].rz(alpha, q[8])
                self.circuits[-1].rz(beta, q[10])
                self.circuits[-1].sx(q[8])
                self.circuits[-1].sx(q[10])
                self.circuits[-1].rz(-alpha, q[8])
                self.circuits[-1].rz(-beta, q[10])
                self.cx0(self.circuits[-1],q[8],q[6])
                self.cx0(self.circuits[-1],q[8],q[7])
                self.cx0(self.circuits[-1],q[10],q[11])
                self.cx0(self.circuits[-1],q[10],q[12])
                self.cx0(self.circuits[-1],q[6],q[4])
                self.cx0(self.circuits[-1],q[7],q[5])
                self.cx0(self.circuits[-1],q[11],q[13])
                self.cx0(self.circuits[-1],q[12],q[14])
                self.cx0(self.circuits[-1],q[4],q[0])
                self.cx0(self.circuits[-1],q[4],q[1])
                self.cx0(self.circuits[-1],q[5],q[2])
                self.cx0(self.circuits[-1],q[5],q[3])
                self.cx0(self.circuits[-1],q[13],q[15])
                self.cx0(self.circuits[-1],q[13],q[16])
                self.cx0(self.circuits[-1],q[14],q[17])
                self.cx0(self.circuits[-1],q[14],q[18])
                self.circuits[-1].measure([q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[15],q[16],q[17],q[18],q[13],q[14],q[11],q[12],q[10]],cr[i])

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
class PB(WitnessJob):
    def __init__(self) -> None:
        super().__init__()

        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
    @staticmethod
    def cx0(c: QuantumCircuit,i,j):
        c.rz(np.pi/2,j)
        c.sx(j)
        c.cz(i,j)
        c.z(j)
        c.sx(j)
        c.rz(np.pi/2,j)



    def add_witness_circuits(self, qubits_list: List[int]) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()

        self.circuits.clear()
        for s in range(32 * self.n_repetitions):
            #self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr=[]
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(5, "cr"+str(i)))
            qreg = QuantumRegister(156)
            #self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q=qubits_list[i]
                par=self.indices_list[i][s]
                self.circuits[-1].sx(q[2])
                self.circuits[-1].s(q[2])
                self.cx0(self.circuits[-1],q[2],q[3])
                self.cx0(self.circuits[-1],q[2],q[1])              
                self.cx0(self.circuits[-1],q[3],q[4])
                self.cx0(self.circuits[-1],q[1],q[0])
                self.cx0(self.circuits[-1],q[3],q[2])
                self.circuits[-1].x(q[2])
                t=np.arccos(np.sqrt(3/5))
                self.circuits[-1].z(q[2])
                self.circuits[-1].s(q[3])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].rzz(t,q[3],q[2])
                self.circuits[-1].sdg(q[3])
                self.circuits[-1].sdg(q[2])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].rzz(t,q[3],q[2])
                self.circuits[-1].sdg(q[3])
                self.circuits[-1].sdg(q[2])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].s(q[2])
                self.circuits[-1].z(q[3])
                t=np.arccos(np.sqrt(1/3))
                #2-1
                self.circuits[-1].z(q[2])
                self.circuits[-1].s(q[1])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].rzz(t,q[1],q[2])
                self.circuits[-1].sdg(q[1])
                self.circuits[-1].sdg(q[2])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].rzz(t,q[1],q[2])
                self.circuits[-1].sdg(q[1])
                self.circuits[-1].sdg(q[2])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[2])
                self.circuits[-1].s(q[2])
                self.circuits[-1].z(q[1])
                #3-4
                t=np.pi/4
                self.circuits[-1].z(q[3])
                self.circuits[-1].s(q[4])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].rzz(t,q[3],q[4])
                self.circuits[-1].sdg(q[3])
                self.circuits[-1].sdg(q[4])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].rzz(t,q[3],q[4])
                self.circuits[-1].sdg(q[3])
                self.circuits[-1].sdg(q[4])
                self.circuits[-1].sx(q[3])
                self.circuits[-1].sx(q[4])
                self.circuits[-1].s(q[3])
                self.circuits[-1].z(q[4])               
                #1-0
                self.circuits[-1].z(q[1])
                self.circuits[-1].s(q[0])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].rzz(t,q[1],q[0])
                self.circuits[-1].sdg(q[1])
                self.circuits[-1].sdg(q[0])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].rzz(t,q[1],q[0])
                self.circuits[-1].sdg(q[1])
                self.circuits[-1].sdg(q[0])
                self.circuits[-1].sx(q[1])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].s(q[1])
                self.circuits[-1].z(q[0])
                for kk in range(5):
                    self.circuits[-1].rz(np.pi/16,q[kk])
                    if par%2==0:
                        self.circuits[-1].s(q[kk])
                    self.circuits[-1].sx(q[kk])
                    par//=2     
                self.circuits[-1].measure([q[0],q[1],q[2],q[3],q[4]],cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(32):
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
        for s in range(32*self.n_repetitions):
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

