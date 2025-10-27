import json
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from tqdm import tqdm


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


class WitnessResult:
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table

    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = self.raw_results._append(result)

    def CalculateMatrix(self) -> np.array:
        self.matrix = np.ones([5, 5])
        for i in range(5):
            for j in range(4):
                selected_row = self.raw_results.loc[
                    (self.raw_results["i"] == i) & (self.raw_results["j"] == j),
                    ["0", "1"],
                ]
                p_0 = selected_row["0"] / (selected_row["0"] + selected_row["1"])
                ij_element = p_0.iloc[0]
                self.matrix[j, i] = ij_element
        return self.matrix

    def SumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["i", "j"], as_index=False).sum()


def matrix_minor(arr, i, j):
    return np.linalg.det(np.delete(np.delete(arr, i, axis=0), j, axis=1))


def adj(A: np.array) -> np.array:
    ad = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if (i + j) % 2:
                ad[i][j] = -matrix_minor(A, j, i)
            else:
                ad[i][j] = matrix_minor(A, j, i)
    return ad


def std_dev(P: np.array, n: float) -> float:
    adj_P = adj(P)
    standard_deviation = 0
    for j in range(len(P)):
        for k in range(len(P)):
            standard_deviation += (adj_P[j][k]) ** 2 * P[k][j] * (1 - P[k][j])
    standard_deviation = standard_deviation / n
    return standard_deviation


def plot_results(qubits, vb, erb, va, er) -> None:
    _, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_xticks(
        [i for i in range(len(qubits))], labels=[f"b{str(q)}" for q in qubits]
    )

    ax.errorbar(
        [i for i in range(len(qubits))],
        vb,
        erb,
        linewidth=0,
        capsize=3,
        elinewidth=1,
        marker="o",
        color="red",
    )

    ax.errorbar(
        [i for i in range(len(qubits))],
        va,
        er,
        linewidth=0,
        capsize=3,
        elinewidth=1,
        marker="o",
        color="blue",
    )

    axis_formatter = ticker.ScalarFormatter(useMathText=True)
    axis_formatter.set_powerlimits((3, 4))
    ax.yaxis.set_major_formatter(axis_formatter)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.get_yaxis().get_offset_text().set_visible(False)
    ax_max = max(ax.get_yticks())
    # print(ax.get_yticks())
    # print(ax_max)
    exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    ax.annotate(
        r"$\times$10$^{%i}$" % (exponent_axis),
        xy=(0.01, 0.91),
        xycoords="axes fraction",
    )
    plt.savefig("dtest-viv-n.pdf", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.show()


def main() -> None:
    results_path = "k:\\Coding\\Python\\IBMQ_Witness_Tests\\simulations\\"

    # fig,ax=plt.subplots(1,1, figsize=(5, 3),tight_layout=True)

    va = []
    vb = []
    er = []
    # dat = [[] for _ in range(4)]
    # eri = [[] for _ in range(4)]
    dat = []
    eri = []
    erb = []

    device = "ibm_brisbane"
    n_qubits = 1
    n_shots = int(1e6)

    qubits = [i for i in range(n_qubits)]
    qubits = [18, 24, 53, 67, 81, 113]

    for ii in tqdm(range(len(qubits))):
        dat.append([])
        eri.append([])

        summed_result = WitnessResult(pd.DataFrame())

        pd_result = pd.read_csv(f"{results_path}{device}_{qubits[ii]}.csv", index_col=0)
        pd_result.rename(columns={"0 0": "0", "1 0": "1"}, inplace=True)
        summed_result.AppendResults(pd_result)
        current_result = WitnessResult(pd_result)
        current_result.SumResults()
        current_result.raw_results
        current_P = current_result.CalculateMatrix()
        dat[ii].append(np.linalg.det(current_P))
        eri[ii].append(std_dev(current_P, n_shots))

        summed_result.SumResults()
        summed_result.raw_results
        P = summed_result.CalculateMatrix()

        print(P)

        va.append(np.linalg.det(P))
        er.append(np.sqrt(std_dev(P, n_shots)))

        print(f"\n{qubits[ii]}: {np.linalg.det(P)} ± {np.sqrt(std_dev(P, n_shots))}")

        vb.append(sum(dat[ii]) / len(dat[ii]))
        erb.append(np.sqrt(sum(eri[ii])) / len(eri[ii]))

    va_abs = [abs(i) for i in va]
    print(f"min: {va_abs.index(min(va_abs))}, max: {va_abs.index(max(va_abs))}")
    plot_results(qubits, vb, erb, va, er)


if __name__ == "__main__":
    main()
