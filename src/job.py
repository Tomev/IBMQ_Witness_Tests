"""
This script stores our Job classes. 
"""

import os
import random
from abc import abstractmethod
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy import pi
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import PrimitiveResult


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

    def add_sanity_test_circuits(self, test_number: int) -> None:
        """
        Sanity check test circuits.
        """
        self.test_circuits_number = test_number

        circuit_test_0 = QuantumCircuit(1, 1)
        circuit_test_0.measure(0, 0)

        circuit_test_1 = QuantumCircuit(1, 1)
        circuit_test_1.x(0)
        circuit_test_1.measure(0, 0)

        for _ in range(0, test_number):
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

        results = self.get_counts_from_job_results(self.queued_job)

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
        except Exception as alert:
            print(alert)

    @staticmethod
    def get_counts_from_job_results(job):
        """
        Get counts from the job.

        Job results, returned by differently executed jobs (eg. by a sampler
        or by a backend) have a different structure. This method is supposed
        to be overwritten in the child classes, to provide a unified way of
        getting counts from the job results.
        """
        # TODO TR:  We might require some try-except block here, expecting
        #           different types of job results.

        results = job.result()
        counts = []

        # Check if type is PrimitiveResult
        if isinstance(results, PrimitiveResult):
            # This is for the case when the job is executed by a sampler.

            # print(f"\n{results}\n")
            # print(f"\n{results[0]}\n")
            # print(f"\n{results[0].data}\n")

            # TODO TR:  This assumes that the register is called "c". That
            #           might not be the case for all the results.

            counts = []

            for result in results:
                counts.append(result.data.c.get_counts())

        else:
            # This is for the case when the job is executed by a backend.
            counts = results.get_counts()

        # print(f"\n{counts}\n")

        return counts


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
        except Exception as alert:
            print(alert)


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
        except Exception as alert:
            print(alert)


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
            self.circuits.append(QuantumCircuit(127, len(qubits_list)))
            # self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests

            for i, qubit in enumerate(qubits_list):
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
                self.circuits[-1].measure(qubit, i)

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
        result_counts = self.get_counts_from_job_results(self.queued_job)
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


class TestJob(Job):
    """
    An abstract class for Test jobs.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def add_test_circuits(self, test_number: int) -> None:
        """
        Adds test circuits to the job.
        """
        raise NotImplementedError

    def save_to_file(self, csv_path, zip_filename):
        result_counts = []

        job_result = self.queued_job.result()
        for pub_result in job_result:
            for i in range(len(self.qubits_list)):
                result_counts.append(
                    getattr(pub_result.data, "cr" + str(i)).get_counts()
                )
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)

        indices_i = []
        indices_q = []

        for s in range(8 * self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva = self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q

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


class Weak(TestJob):
    def __init__(self) -> None:
        super().__init__()
        self.ep = 0
        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []

    @staticmethod
    def we(c: QuantumCircuit, i, j, eps):
        c.ecr(i, j)
        c.rz(eps, j)
        c.ecr(i, j)
        c.rz(np.pi / 2, j)
        c.sx(j)

    def add_test_circuits(self, qubits_list: List[int], epp) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()
        self.ep = epp

        self.circuits.clear()
        for s in range(8 * self.n_repetitions):
            # self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr = []
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr" + str(i)))
            qreg = QuantumRegister(156)
            # self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q = qubits_list[i]
                par = self.indices_list[i][s]
                a = par % 2
                b = (par // 2) % 2
                c = par // 4
                alpha = (2 * a - 1) * epp
                beta = (2 * b - 1) * epp

                # TODO TR: Refactor that if.

                if c:
                    self.we(self.circuits[-1], q[0], q[1], alpha)
                    self.circuits[-1].rz(np.pi / 2, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2, q[0])
                    self.we(self.circuits[-1], q[0], q[2], beta)
                    self.circuits[-1].rz(-np.pi / 2, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2, q[0])
                else:
                    self.circuits[-1].rz(np.pi / 2, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2, q[0])
                    self.we(self.circuits[-1], q[0], q[2], beta)
                    self.circuits[-1].rz(-np.pi / 2, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2, q[0])
                    self.we(self.circuits[-1], q[0], q[1], alpha)

                self.circuits[-1].rz(np.pi / 2, q[0])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].rz(-np.pi / 2, q[0])
                self.circuits[-1].measure([q[0], q[1], q[2]], cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(8):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)


class LG(TestJob):

    def __init__(self) -> None:
        super().__init__()
        self.ep = 0
        self.indices_list = []
        self.n_repetitions = 1
        self.qubits_list = []
        self.qubits_dir = []

    @staticmethod
    def we(c: QuantumCircuit, i, j, eps):

        c.ecr(i, j)
        c.rz(eps, j)
        c.ecr(i, j)

        # Y_-
        c.rz(np.pi / 2, j)
        c.sx(j)
        # We can skip the RZ rotation before the measurememt

    def add_test_circuits(self, qubits_list: List[int], epp) -> None:
        self.qubits_list = qubits_list
        self._get_angles_lists()
        self.ep = epp

        self.circuits.clear()

        for s in range(8 * self.n_repetitions):
            # self.circuits.append(QuantumCircuit(127, len(listvert)))
            cr = []
            for i in range(len(qubits_list)):
                cr.append(ClassicalRegister(3, "cr" + str(i)))
            qreg = QuantumRegister(156)
            # self.circuits.append(QuantumCircuit(2, len(qubits_list)))  # TR: For tests
            self.circuits.append(QuantumCircuit(qreg, *cr))
            for i in range(len(qubits_list)):
                q = qubits_list[i]

                # print(self.indices_list)

                par = self.indices_list[i][s]

                # print(par)

                # Zdaje się, że a, b, c to flagi sterujące eksperymentem.
                a = par % 2  # Liczenie kąta pomiaru na qubicie a
                b = (par // 2) % 2  # Liczenie kąta pomiaru na qubicie b
                c = par // 4  # Kolejność pomiarów słabych

                # print(f"a={a}, b={b}, c={c}")

                # Wartości kątów w zależności od flagi. (-epp lub epp)
                alpha = (2 * a - 1) * epp  # Kąt pomiaru na qubicie a
                beta = (2 * b - 1) * epp  # Kąt pomiaru na qubicie b

                aa = np.pi / 4
                bb = -np.pi / 4

                # Y_+ |0> = 1/sqrt(2) (|0> + |1>) state
                self.circuits[-1].rz(-np.pi / 2, q[0])
                self.circuits[-1].sx(q[0])
                self.circuits[-1].rz(np.pi / 2, q[0])

                # TODO TR: Refactor that if.
                if c:
                    self.circuits[-1].rz(np.pi / 2 + aa, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2 - aa, q[0])
                    self.we(self.circuits[-1], q[0], q[1], alpha)
                    self.circuits[-1].rz(-np.pi / 2 + aa, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2 - aa, q[0])

                    self.circuits[-1].rz(np.pi / 2 + bb, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2 - bb, q[0])
                    self.we(self.circuits[-1], q[0], q[2], beta)
                    self.circuits[-1].rz(-np.pi / 2 + bb, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2 - bb, q[0])
                else:
                    self.circuits[-1].rz(np.pi / 2 + bb, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2 - bb, q[0])
                    self.we(self.circuits[-1], q[0], q[2], beta)
                    self.circuits[-1].rz(-np.pi / 2 + bb, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2 - bb, q[0])
                    self.circuits[-1].rz(np.pi / 2 + aa, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(-np.pi / 2 - aa, q[0])
                    self.we(self.circuits[-1], q[0], q[1], alpha)
                    self.circuits[-1].rz(-np.pi / 2 + aa, q[0])
                    self.circuits[-1].sx(q[0])
                    self.circuits[-1].rz(np.pi / 2 - aa, q[0])

                # Prepare initial qubit measurement in Y basis.
                self.circuits[-1].sx(q[0])
                self.circuits[-1].measure([q[0], q[1], q[2]], cr[i])

    def _get_angles_lists(self):
        for v in self.qubits_list:
            self.va = []
            for n in range(self.n_repetitions):
                for i in range(8):
                    self.va.append(i)
            random.shuffle(self.va)
            self.indices_list.append(self.va)

    def save_to_file(self, csv_path, zip_filename):
        result_counts = []
        job_result = self.queued_job.result()
        for pub_result in job_result:
            for i in range(len(self.qubits_list)):
                result_counts.append(
                    getattr(pub_result.data, "cr" + str(i)).get_counts()
                )
        pandas_table = pd.DataFrame.from_dict(result_counts).fillna(0)
        indices_i = []
        indices_q = []
        # qubits_list=self.qubits_list
        for s in range(8 * self.n_repetitions):
            for q in range(len(self.qubits_list)):
                iva = self.indices_list[q][s]
                indices_i.append(iva)
                indices_q.append(q)
        pandas_table["i"] = indices_i
        pandas_table["q"] = indices_q

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


class LGSingleGate(LG):

    def __init__(self):
        super().__init__()

    @staticmethod
    def we(c: QuantumCircuit, i, j, eps):
        c.sx(j)
        c.rz(eps + np.pi / 2, j)
        c.ecr(i, j)
        c.rz(np.pi / 2, i)
        c.x(i)

class LGZZ(LG):

    def __init__(self):
        super().__init__()

    @staticmethod
    def we(c: QuantumCircuit, i, j, eps):   
        if eps>=0:
            c.sx(j) 
            c.rzz(eps,  i, j)
            c.rz(np.pi/2,j)
            c.sx(j)
        else:
            c.sx(j)
            c.rzz(-eps,  i, j)
            c.rz(-np.pi/2,j)
            c.sx(j)