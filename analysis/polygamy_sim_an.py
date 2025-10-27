import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils import ExperimentSetup


class PolygamyResults:
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table

    def appendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = self.raw_results = pd.concat(
            [self.raw_results, pd.DataFrame(result)], ignore_index=True
        )

    def calculate(self, set_number: int):
        b = []
        for i in range(32):
            a = [0 for _ in range(32)]
            selected_row = self.raw_results.loc[
                (self.raw_results["q"] == set_number) & (self.raw_results["i"] == i)
            ]
            for ke, va in selected_row.items():
                if len(ke) == 5:
                    z = 0
                    j = 1
                    for c in ke[::-1]:
                        if c == "1":
                            z += j
                        j <<= 1

                    a[z] = va.to_numpy()[0]
            b.append(a)
        return b

    def sumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["i", "q"], as_index=False).sum()


def prepare_results(results_path: str, exp_setup: ExperimentSetup) -> PolygamyResults:
    results = PolygamyResults(pd.DataFrame())

    file_path: str = f"{results_path}/{exp_setup.backend}_{exp_setup.qubit_group}.csv"

    pd_result = pd.read_csv(file_path, index_col=0)

    results.appendResults(pd_result)
    results.sumResults()

    return results


def analyze_polygamy_results(results_path: str, setups: List[ExperimentSetup]):
    minimal_violations: List[Tuple[Tuple[int, ...], float]] = []

    for setup in setups:
        results: PolygamyResults = prepare_results(results_path, setup)

        print(setup)

        b = results.calculate(0)
        n = 2 * sum(b[0])
        print("Trials", n // 2)
        # minimal M
        MM = 10
        for j in range(5):
            # xxxx
            ee = []
            sig = 0
            e = 0
            pj = 1 << j
            aa = [[0] * 16] * 16

            for k in range(32):
                for m in range(32):
                    aa[m % pj + ((m // pj) // 2) * pj][
                        k % pj + ((k // pj) // 2) * pj
                    ] += b[m][k]

            p = [0, 3, 5, 9, 6, 10, 12, 15]
            r = [1, 2, 4, 7, 8, 11, 13, 14]
            xxp = 0
            xxm = 0

            for w in p:
                xxp += aa[0][w]
            xxm = 0

            for w in r:
                xxm += aa[0][w]

            sig += xxp * xxm / n**2
            ee.append(xxp - xxm)
            e += xxp - xxm
            v = [3, 5, 9, 6, 10, 12]

            for g in v:
                xyp = 0
                xym = 0
                for w in p:
                    xyp += aa[g][w]

                for w in r:
                    xym += aa[g][w]
                sig += xyp * xym / n**2
                ee.append(-xyp + xym)
                e += -xyp + xym
            yyp = 0

            for w in p:
                yyp += aa[15][w]
            yym = 0

            for w in r:
                yym += aa[15][w]
            sig += yyp * yym / n**2
            ee.append(yyp - yym)
            e += yyp - yym
            v = [1, 2, 4, 8]

            for g in v:
                xyp = 0
                xym = 0
                for w in p:
                    xyp += aa[g][w]

                for w in r:
                    xym += aa[g][w]
                sig += xyp * xym / n**2
                ee.append(xyp - xym)
                e += xyp - xym

            v = [14, 13, 11, 7]

            for g in v:
                xyp = 0
                xym = 0
                for w in p:
                    xyp += aa[g][w]

                for w in r:
                    xym += aa[g][w]
                sig += xyp * xym / n**2
                ee.append(-xyp + xym)
                e += -xyp + xym

            print("Mermin correlations excl", j)

            for k in range(16):
                print(ee[k] / n, end=" ")

            print()
            print("M = ", e / n, "+-", 2 * np.sqrt(sig / n))
            MM = min(e / n, MM)
        print("Minimal violation:", MM)
        minimal_violations.append((setup.qubit_group, MM))

    minimal_violations.sort(key=lambda x: x[1], reverse=True)
    analysis_results_file_path: str = f"{setups[0].backend}_polygamy_results.csv"
    save_to_file(analysis_results_file_path, minimal_violations)

    greedy_nonoverlapping_maximal_minimal_violations: List[
        Tuple[Tuple[int, ...], float]
    ] = find_greedy_nonoverlapping_maximal_minimal_violations(minimal_violations)
    analysis_results_file_path = (
        f"{setups[0].backend}_greedy_nonoverlaping_polygamy_results.csv"
    )
    save_to_file(
        analysis_results_file_path, greedy_nonoverlapping_maximal_minimal_violations
    )


def find_greedy_nonoverlapping_maximal_minimal_violations(
    sorted_minimal_violations: List[Tuple[Tuple[int, ...], float]],
) -> List[Tuple[Tuple[int, ...], float]]:
    nonoverlapping_violations: List[Tuple[Tuple[int, ...], float]] = []

    qubits_used: List[int] = []

    for group, violation in sorted_minimal_violations:
        if not any(q in qubits_used for q in group):
            qubits_used.extend(group)
            nonoverlapping_violations.append((group, violation))

    return nonoverlapping_violations


def save_to_file(
    file_path: str, minimal_violations: List[Tuple[Tuple[int, ...], float]]
) -> None:
    with open(file_path, "w") as f:
        f.write("Qubit Group, Minimal Violation\n")
        for group, violation in minimal_violations:
            f.write(f"{str(group).replace(',', ';')}, {violation}\n")


def prepare_setups(results_path: str, backend: str) -> List[ExperimentSetup]:
    setups: List[ExperimentSetup] = []

    # Iterate through all files in the results directory.
    for file_name in os.listdir(results_path):
        if not backend in file_name:
            continue

        # IBM device names are ibm_<city>. Making the file names:
        # ibm_<city>_<qubit_group>.csv
        q_group: str = file_name.split("(")[-1].replace(").csv", "")
        qubit_group: Tuple[int] = tuple([int(q) for q in q_group.split(",")])
        setups.append(ExperimentSetup(backend, qubit_group))

    return setups


if __name__ == "__main__":
    backends: List[str] = ["ibm_pittsburgh"]

    results_path: str = "polygamy_results/"

    for backend in backends:
        setups: List[ExperimentSetup] = prepare_setups(results_path, backend)
        analyze_polygamy_results(results_path, setups)
