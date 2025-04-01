import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import json
from zipfile import ZipFile
from os import environ
from qiskit_ibm_runtime import QiskitRuntimeService


def matrix_minor(arr, i, j):
    return np.linalg.det(np.delete(np.delete(arr,i,axis=0), j, axis=1))        
def adj(A: np.array) -> np.array:
    ad=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if (i+j)%2:
                ad[i][j]=-matrix_minor(A,j,i)
            else:
                ad[i][j]=matrix_minor(A,j,i)
    return ad
def std_dev(P: np.array, n: float) -> float:
    adj_P = adj(P)
    standard_deviation = 0
    staux=0
    for j in range(len(P)):
        for k in range(len(P)):
            #print((adj_P[j][k])**2 * P[k][j] * (1 - P[k][j]))
            standard_deviation += (adj_P[j][k])**2 * P[k][j]* (1 - P[k][j])
            #staux+= adj_P[j][k] * P[k][j]
    #standard_deviation-=staux*staux
            #print(standard_deviation)
    standard_deviation = standard_deviation/n
    return standard_deviation

class BellResult:        
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table
        self.qq=17
        
    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = self.raw_results = pd.concat([self.raw_results,pd.DataFrame(result)], ignore_index=True)
        
        
        
    def Calculate(self):
        b=0
        selected_row = self.raw_results
        b+=selected_row["0"].sum()+selected_row["1"].sum()
        return b/25/self.qq
        
    def SumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["i","q"], as_index = False).sum()
    def CalculateMatrix(self,a) -> np.array:
        self.matrix = np.zeros([5,5])
        #print(su)
        for i in range(5):
            for j in range(5):
                self.matrix[i,j]=0
                #print(i,j)
                #print(self.raw_results)
                selected_row = self.raw_results.loc[(self.raw_results["i"] == i+j*5)  & (self.raw_results["q"] == a)]
                sm=0
                su=selected_row["0"].sum()+selected_row["1"].sum()
                if "0" in selected_row.keys():
                    sm=selected_row["0"].sum()
                    #print(sm)
                self.matrix[i,j]=sm/su
        return self.matrix


            

# Job preparation
qbr=[[94, 95, 90],[6, 5, 7], [58, 71, 59], [62, 61, 63], [52, 37, 56], [116, 115, 117], [50, 49, 51], [21, 20, 22], [125, 124, 126], [108, 112, 107]]
qsh=[[49, 50, 48], [107, 106, 108], [119, 118, 120], [45, 46, 44], [69, 68, 70], [59, 60, 58], [89, 88, 74], [114, 109, 115], [26, 25, 27], [80, 79, 81]]
qky=[[53, 60, 41], [114, 115, 113], [0, 1, 14], [68, 69, 55], [26, 16, 25], [4, 3, 15], [103, 102, 104], [82, 83, 81], [54, 45, 64], [33, 20, 39]]
qq=[qbr,qsh,qky]

TOKEN_VARIABLES=['IBMQ_Token_MS']
TOKENS = {key: environ[key] for key in TOKEN_VARIABLES}

service = QiskitRuntimeService("ibm_quantum",token=TOKENS[TOKEN_VARIABLES[0]])
br= service.backend("ibm_brisbane")
sh= service.backend("ibm_sherbrooke")
ky= service.backend("ibm_kyiv")
ba=[br,sh,ky]
ers=[[[0,0] for i in range(10)] for j in range(3)]
era=[[[0,0,0] for i in range(10)] for j in range(3)]
for x in range(3):
    props=ba[x].properties().to_dict()
    gates=props["gates"]
    qubs=props["qubits"]
    for k in range(10):
        for j in range(3):
            for s in qubs[qq[x][k][j]]:
                if s["name"]=='readout_error':
                    era[x][k][j]=s["value"]
    for g in gates:
        qg=g["qubits"]
        if len(qg)>1:
            for k in range(10):
                if qg[0]==qq[x][k][0]:
                    if qg[1]==qq[x][k][1]:
                        ers[x][k][0]=g["parameters"][0]["value"]
                    elif qg[1]==qq[x][k][2]:
                        ers[x][k][1]=g["parameters"][0]["value"]
                
nam=["brisbane","sherbrooke","kyiv"]
for x in range(3):
    print(nam[x],end="&&&&&&&&\\\\")
    print()
    for d in range(10):
        #print(qubits_list[d][0],"-",qubits_list[d][1],"-",qubits_list[d][2],sep="",end='&')
        print(f'{d}',end="&")
        print(f'{qq[x][d][0]}',end="&")
        print(f'{qq[x][d][1]}',end="&")
        print(f'{qq[x][d][2]}',end="&")
        print(f'{ers[x][d][0]*100:.2g}',end="&")
        print(f'{ers[x][d][1]*100:.2g}',end="&")
        print(f'{era[x][d][0]*100:.2g}',end="&")
        print(f'{era[x][d][1]*100:.2g}',end="&")
        print(f'{era[x][d][2]*100:.2g}',end="\\\\")
        print()
    print("\\midrule")



