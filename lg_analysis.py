import json
from math import sin, sqrt
import pandas as pd

# Settings
n_qubit_sets = 10
n_jobs = 60
results_path = "lgy/results-ky"
states_order = ["000", "100", "010", "110", "001", "101", "011", "111"]
qubits_list = [[94, 95, 90],[6, 5, 7], [58, 71, 59], [62, 61, 63], [52, 37, 56], [116, 115, 117], [50, 49, 51], [21, 20, 22], [125, 124, 126], [108, 112, 107]]


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

    weak_meas_rotation_angle_v = 0.1  # That's our weak measurement rotation angle.
    weak_meas_rotation_angle = sin(weak_meas_rotation_angle_v)

    inequality_values_ab = []
    inequality_values_ba = []
    inequality_values_axc = []
    inequality_values_xac = []
    inequality_values_bxc = []
    inequality_values_xbc = []
    qubs=[]
   # inequality_values_mean = []

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
        c = []
        print("xxC")

        for steering_bit in range(8):

            # TR:   What is s? 000 + 110 + 101 + 011 - 111 - 100 - 010 - 001
            #       Do I add counts multiplied by -1 for every 1 in the state?
            #       If I remember correctly that's exactly it. Especially since ss is used
            #       to calculate ABC and BAC.
            sc = (
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
                + counts_per_steering_bit[steering_bit][states_order.index("100")]
                + counts_per_steering_bit[steering_bit][states_order.index("010")]
                + counts_per_steering_bit[steering_bit][states_order.index("110")]
                - counts_per_steering_bit[steering_bit][states_order.index("001")]
                - counts_per_steering_bit[steering_bit][states_order.index("101")]
                - counts_per_steering_bit[steering_bit][states_order.index("011")]
                - counts_per_steering_bit[steering_bit][states_order.index("111")]
            )
            s = (
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
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
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
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
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
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
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
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
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
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
                +counts_per_steering_bit[steering_bit][states_order.index("000")]
                - counts_per_steering_bit[steering_bit][states_order.index("100")]
                + counts_per_steering_bit[steering_bit][states_order.index("010")]
                - counts_per_steering_bit[steering_bit][states_order.index("110")]
                + counts_per_steering_bit[steering_bit][states_order.index("001")]
                - counts_per_steering_bit[steering_bit][states_order.index("101")]
                + counts_per_steering_bit[steering_bit][states_order.index("011")]
                - counts_per_steering_bit[steering_bit][states_order.index("111")]
            )
            c.append(sc)
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
            #print(sum(counts_per_steering_bit[steering_bit][:4]) / n_shots)

        # Indices in the equations below denote the value of steering_bits of the run.
        # All the ss, ac, ab, bc, aa, bb have 8 elements.

        # For steering_qubit in [0, 3], the weak measurements order is A B, hence
        # we only use 0-3 indices in the equations below.
        print("(BA)C")
        cba= (c[0] + c[3] + c[2] + c[1]) / (
            n_shots * 4
        )
        ec1=4-(c[0]**2 + c[3]**2 + c[2]**2 + c[1]**2)/(n_shots)**2
        print(cba,sqrt(ec1/n_shots)/4)
        print("(AB)C")
        cab= (c[4] + c[7] + c[6] + c[5]) / (
            n_shots * 4
        )
        ec2=4-(c[4]**2 + c[7]**2 + c[6]**2 + c[5]**2)/(n_shots)**2
        print(cab,sqrt(ec2/n_shots)/4)
        print("BAC")
        BAC = (ss[0] + ss[3] - ss[2] - ss[1]) / (
            n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4
        )
        eBAC=4-(ss[0]**2 + ss[3]**2 + ss[2]**2 + ss[1]**2)/(n_shots)**2
        print(BAC,sqrt(eBAC/n_shots)/(4 * weak_meas_rotation_angle * weak_meas_rotation_angle))
        print("ABC")
        ABC = (ss[4] + ss[7] - ss[6] - ss[5]) / (
            n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4
        )
        eABC=4-(ss[4]**2 + ss[7]**2 + ss[6]**2 + ss[5]**2)/(n_shots)**2
        print(ABC,sqrt(eABC/n_shots)/(4 * weak_meas_rotation_angle * weak_meas_rotation_angle))
        print("BA")
        BA = (ab[0] + ab[3] - ab[2] - ab[1]) / (
            n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4
        )
        eBA=4-(ab[0]**2 + ab[3]**2 + ab[2]**2 + ab[1]**2)/(n_shots)**2
        print(BA,sqrt(eBA/n_shots)/(4 * weak_meas_rotation_angle * weak_meas_rotation_angle))
        print("AB")
        AB = (ab[4] + ab[7] - ab[6] - ab[5]) / (
            n_shots * weak_meas_rotation_angle * weak_meas_rotation_angle * 4
        )
        eAB=4-(ab[4]**2 + ab[7]**2 + ab[6]**2 + ab[5]**2)/(n_shots)**2
        print(AB,sqrt(eAB/n_shots)/(4 * weak_meas_rotation_angle * weak_meas_rotation_angle))
        print("xAC")
        xAC = -(ac[0] - ac[3] + ac[2] - ac[1]) / (n_shots * weak_meas_rotation_angle * 4)
        exAC=4-(ac[0]**2 + ac[3]**2 + ac[2]**2 + ac[1]**2)/(n_shots)**2
        print(xAC,sqrt(exAC/n_shots)/(4 * weak_meas_rotation_angle))
        print("AXC")
        AxC = -(ac[4] - ac[7] + ac[6] - ac[5]) / (n_shots * weak_meas_rotation_angle * 4)
        eAxC=4-(ac[4]**2 + ac[7]**2 + ac[6]**2 + ac[5]**2)/(n_shots)**2
        print(AxC,sqrt(eAxC/n_shots)/(4 * weak_meas_rotation_angle))
        print("BxC")
        BxC = -(bc[0] - bc[3] - bc[2] + bc[1]) / (n_shots * weak_meas_rotation_angle * 4)
        eBxC=4-(bc[0]**2 + bc[3]**2 + bc[2]**2 + bc[1]**2)/(n_shots)**2
        print(BxC,sqrt(eBxC/n_shots)/(4 * weak_meas_rotation_angle)) 
        print("xBC")
        xBC = -(bc[4] - bc[7] - bc[6] + bc[5]) / (n_shots * weak_meas_rotation_angle * 4)
        exBC=4-(bc[4]**2 + bc[7]**2 + bc[6]**2 + bc[5]**2)/(n_shots)**2
        print(xBC,sqrt(exBC/n_shots)/(4 * weak_meas_rotation_angle))
        print("xA")
        xA = -(aa[0] - aa[3] + aa[2] - aa[1]) / (n_shots * weak_meas_rotation_angle * 4)
        exA=4-(aa[0]**2 + aa[3]**2 + aa[2]**2 + aa[1]**2)/(n_shots)**2
        print(xA,sqrt(exA/n_shots)/(4 * weak_meas_rotation_angle))
        print("Ax")
        Ax = -(aa[4] - aa[7] + aa[6] - aa[5]) / (n_shots * weak_meas_rotation_angle * 4)
        eAx=4-(aa[4]**2 + aa[7]**2 + aa[6]**2 + aa[5]**2)/(n_shots)**2
        print(Ax,sqrt(eAx/n_shots)/(4 * weak_meas_rotation_angle))
        print("Bx")
        Bx = -(bb[0] - bb[3] - bb[2] + bb[1]) / (n_shots * weak_meas_rotation_angle * 4)
        eBx=4-(bb[0]**2 + bb[3]**2 + bb[2]**2 + bb[1]**2)/(n_shots)**2
        print(Bx,sqrt(eBx/n_shots)/(4 * weak_meas_rotation_angle))
        print("xB")
        xB = -(bb[4] - bb[7] - bb[6] + bb[5]) / (n_shots * weak_meas_rotation_angle * 4)
        exB=4-(bb[4]**2 + bb[7]**2 + bb[6]**2 + bb[5]**2)/(n_shots)**2
        print(xB,sqrt(exB/n_shots)/(4 * weak_meas_rotation_angle))
        qubs.append(qubits_list[qubit_set_idx])
        inequality_values_ba.append(Bx + xA - BA)
        inequality_values_ab.append(Ax + xB - AB)
        inequality_values_bxc.append(Bx - cba + BxC)
        inequality_values_xbc.append(xB - cab + xBC)
        inequality_values_xac.append(xA + cba - xAC)
        inequality_values_axc.append(xA + cab - AxC)
        

    #inequality_values_ab.sort(reverse=True, key=lambda x: x[1])
    #inequality_values_ba.sort(reverse=True, key=lambda x: x[1])
    #inequality_values_mean.sort(reverse=True, key=lambda x: x[1])

    indices = range(len(inequality_values_ab))
    df = pd.DataFrame(
        columns=["qubits","ba", "ab", "bxc","xbc","xac","axc"], index=indices
    )

    for i in indices:
        df.loc[i, "qubits"]=qubs[i]
        df.loc[i, "ba"] = inequality_values_ba[i]
        df.loc[i, "ab"] = inequality_values_ab[i]
        df.loc[i, "bxc"] = inequality_values_bxc[i]
        df.loc[i, "xbc"] = inequality_values_xbc[i]
        df.loc[i, "xac"] = inequality_values_xac[i]
        df.loc[i, "axc"] = inequality_values_axc[i]

    df.to_csv(f"{results_path}/results_kyiv_real.csv")


if __name__ == "__main__":
    main()
