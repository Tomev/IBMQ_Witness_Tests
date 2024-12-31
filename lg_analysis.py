import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import json
from zipfile import ZipFile

# Settings
n_qubit_sets = 13
n_jobs = 60
results_path = "lg/results"
states_order = ["000", "100", "010", "110", "001", "101", "011", "111"]


def load_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    metadata = {
        "jobs": json_data["jobs"],  # ile jobów wykonujemy
        "shots": json_data["shots"],  # ile shotów dla każdego układu
        "repetitions": json_data["repetitions"],  # ile shotów dla każdego układu
        "randomization": json_data[
            "randomization"
        ],  # randomizacja kątów włączona/wyłączona
        "backend": json_data["backend"],
    }

    return metadata


# from JobResult import *


class BellResult:
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table

    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = pd.concat(
            [self.raw_results, pd.DataFrame(result)], ignore_index=True
        )

    def Calculate(self, qubit_set_idx):
        b = []
        for steering_bit in range(8):
            a = []
            selected_row = self.raw_results.loc[
                (self.raw_results["q"] == qubit_set_idx)
                & (self.raw_results["i"] == steering_bit)
            ]
            for state in states_order:
                a.append(selected_row[state].to_numpy()[0])

            b.append(a)

        return b

    def SumResults(self) -> None:
        # There are 13 qubit grups and 8 steering bits
        # We expect 13*8=104 rows in the table.
        # q denotes qubit_set_index and i steering_bit "value"
        self.raw_results = self.raw_results.groupby(["i", "q"], as_index=False).sum()


def main():
    # metadata = load_json(results_path + '/data.json')
    summed_result = BellResult(pd.DataFrame())

    for i in range(n_jobs):  # metadata["jobs"]):
        pd_result = pd.read_csv(
            results_path + "/results_tests_" + str(i) + ".csv", index_col=0
        )

        summed_result.AppendResults(pd_result)

        raw_results = BellResult(pd.DataFrame())
        raw_results.AppendResults(pd_result)

        # print("\n\n\n")
        # print(raw_results.raw_results)

        raw_results.SumResults()

        # print("\n\n\n")
        # print(raw_results.raw_results)

        # TR:   Commented code block below seems to compute n_shots for each qubit_set.
        #       It resets n_shots variable each time, so I don't think it's useful.
        #       Commenting it didn't change the results.
        """ 
        for qubit_set_idx in range(n_qubit_sets):
            counts_per_steering_bit = raw_results.Calculate(qubit_set_idx)
            
            n_shots = 0

            # print(len(counts_per_steering_bit))
            # print(counts_per_steering_bit)
            
            for measured_state_idx in range(8):
                n_shots += counts_per_steering_bit[0][measured_state_idx]
        """

    summed_result.SumResults()

    weak_meas_rotation_angle = 0.1  # That's our weak measurement rotation angle.

    inequality_values_per_set = []

    for qubit_set_idx in range(n_qubit_sets):

        counts_per_steering_bit = summed_result.Calculate(qubit_set_idx)

        n_shots = 0

        for measured_state_idx in range(8):
            n_shots += counts_per_steering_bit[0][measured_state_idx]

        print("Qubit_set_index:", qubit_set_idx)
        print("Trials: ", n_shots)

        # print(counts_per_steering_bit)

        ss = []
        ac = []
        ab = []
        bc = []
        aa = []
        bb = []

        print("xxC")

        for steering_bit in range(8):

            # TR:   What is s? 000 + 110 + 101 + 011 - 111 - 100 - 010 - 001
            #       Do I add counts multiplied by -1 for every 1 in the state?
            #       If I remember correctly that's exactly it. Especially since ss is used
            #       to calculate ABC and BAC.
            s = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                - counts_per_steering_bit[steering_bit][states_order.index("100")]
                - counts_per_steering_bit[steering_bit][states_order.index("010")]
                + counts_per_steering_bit[steering_bit][states_order.index("110")]
                - counts_per_steering_bit[steering_bit][states_order.index("001")]
                + counts_per_steering_bit[steering_bit][states_order.index("101")]
                + counts_per_steering_bit[steering_bit][states_order.index("011")]
                - counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            # TR:   What is sac? 000 + 100 + 011 + 111 - 010 - 110 - 001 - 101
            #       I add whenever there are the same values for qubits 0 and 1.
            sac = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                + counts_per_steering_bit[steering_bit][states_order.index("100")]
                - counts_per_steering_bit[steering_bit][states_order.index("010")]
                - counts_per_steering_bit[steering_bit][states_order.index("110")]
                - counts_per_steering_bit[steering_bit][states_order.index("001")]
                - counts_per_steering_bit[steering_bit][states_order.index("101")]
                + counts_per_steering_bit[steering_bit][states_order.index("011")]
                + counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            # TR:   What is sbc? 000 + 010 + 101 + 111 - 100 - 110 - 001 - 011
            #       I add whenever there are the same values for qubits 0 and 2.
            sbc = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                - counts_per_steering_bit[steering_bit][states_order.index("100")]
                + counts_per_steering_bit[steering_bit][states_order.index("010")]
                - counts_per_steering_bit[steering_bit][states_order.index("110")]
                - counts_per_steering_bit[steering_bit][states_order.index("001")]
                + counts_per_steering_bit[steering_bit][states_order.index("101")]
                - counts_per_steering_bit[steering_bit][states_order.index("011")]
                + counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            # TR:   What is sab? 000 + 001 + 110 + 111 - 100 - 101 - 011 - 010
            #       I add whenever there are the same values for qubits 1 and 2.
            sab = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                - counts_per_steering_bit[steering_bit][states_order.index("100")]
                - counts_per_steering_bit[steering_bit][states_order.index("010")]
                + counts_per_steering_bit[steering_bit][states_order.index("110")]
                + counts_per_steering_bit[steering_bit][states_order.index("001")]
                - counts_per_steering_bit[steering_bit][states_order.index("101")]
                - counts_per_steering_bit[steering_bit][states_order.index("011")]
                + counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            # TR:   What is sa? 000 + 100 + 001 + 101 - 010 - 011 - 110 - 111
            #       Add every state with 0 on qubit 1.
            sa = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                + counts_per_steering_bit[steering_bit][states_order.index("100")]
                - counts_per_steering_bit[steering_bit][states_order.index("010")]
                - counts_per_steering_bit[steering_bit][states_order.index("110")]
                + counts_per_steering_bit[steering_bit][states_order.index("001")]
                + counts_per_steering_bit[steering_bit][states_order.index("101")]
                - counts_per_steering_bit[steering_bit][states_order.index("011")]
                - counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            # TR:   What is sb? 000 + 001 + 010 + 011 - 100 - 101 - 110 - 111
            #       Add every state with 0 on qubit 2.
            sb = (
                + counts_per_steering_bit[steering_bit][states_order.index("000")]
                - counts_per_steering_bit[steering_bit][states_order.index("100")]
                + counts_per_steering_bit[steering_bit][states_order.index("010")]
                - counts_per_steering_bit[steering_bit][states_order.index("110")]
                + counts_per_steering_bit[steering_bit][states_order.index("001")]
                - counts_per_steering_bit[steering_bit][states_order.index("101")]
                + counts_per_steering_bit[steering_bit][states_order.index("011")]
                - counts_per_steering_bit[steering_bit][states_order.index("111")]
            )

            ss.append(s)
            ac.append(sac)
            ab.append(sab)
            bc.append(sbc)
            aa.append(sa)
            bb.append(sb)

            # print(*b[j])
            
            # print(counts_per_steering_bit[steering_bit])
            # print(counts_per_steering_bit[steering_bit][:4])

            # Prints sum of counts with 0 on the right bit (XX0).
            print(sum(counts_per_steering_bit[steering_bit][:4]) / n_shots)
            
        # Indices in the equations below denote the value of steering_bits of the run.
        # All the ss, ac, ab, bc, aa, bb have 8 elements. 
        
        # For steering_qubit in [0, 3], the weak measurements order is A B, hence 
        # we only use 0-3 indices in the equations below. 
        print("ABC")
        ABC = (ss[0] + ss[3] - ss[2] - ss[1])/ (n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4)
        print(
            ABC
        )
        print("BAC")
        BAC = (ss[4] + ss[7] - ss[6] - ss[5])/ (n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4)
        print(
            BAC
        )
        print("AB")
        AB = (ab[0] + ab[3] - ab[2] - ab[1]) / (n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4) 
        print(
            AB
        )
        print("BA")
        BA = (ab[4] + ab[7] - ab[6] - ab[5]) / (n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4) 
        print(
            BA
        )
        print("AxC")
        AxC = (ac[0] - ac[3] + ac[2] - ac[1]) / (n_shots * weak_meas_rotation_angle * 4)
        print(
            AxC
            
        )
        print("xAC")
        xAC = (ac[4] - ac[7] + ac[6] - ac[5]) / (n_shots * weak_meas_rotation_angle * 4)
        print(
            xAC
        )
        print("xBC")
        xBC = (bc[0] - bc[3] - bc[2] + bc[1]) / (n_shots * weak_meas_rotation_angle * 4)
        print(
            xBC
        )
        print("BxC")
        BxC = (bc[4] - bc[7] - bc[6] + bc[5]) / (n_shots * weak_meas_rotation_angle * 4)
        print(
            BxC
        )
        print("Ax")
        Ax = (aa[0] - aa[3] + aa[2] - aa[1]) / (n_shots * weak_meas_rotation_angle * 4)
        print(
            Ax
        )
        print("xA")
        xA = (aa[4] - aa[7] + aa[6] - aa[5]) / (n_shots * weak_meas_rotation_angle * 4) 
        print(
            xA
        )
        print("xB")
        xB = (bb[0] - bb[3] - bb[2] + bb[1]) / (n_shots * weak_meas_rotation_angle * 4) 
        print(
            xB
        )
        print("Bx")
        Bx = (bb[4] - bb[7] - bb[6] + bb[5]) / (n_shots * weak_meas_rotation_angle * 4) 
        print(
            Bx
        )

        inequality_values_per_set.append(
            Bx + xA - BA
        )

    print(inequality_values_per_set)

if __name__ == "__main__":
    main()
