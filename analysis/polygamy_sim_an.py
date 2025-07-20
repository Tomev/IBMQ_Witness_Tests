import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from utils import ExperimentSetup



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


def analyze_polygamy_results(results_path: str, setup: ExperimentSetup):

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
                aa[m % pj + ((m // pj) // 2) * pj][k % pj + ((k // pj) // 2) * pj] += b[
                    m
                ][k]

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


if __name__ == "__main__":
    setups: List[ExperimentSetup] = [
        ExperimentSetup("noiseless_simulator", [0, 1, 2, 3, 4])
    ]
    results_path: str = "results/"

    for setup in setups:
        analyze_polygamy_results(results_path, setup)
