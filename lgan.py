import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import json
from zipfile import ZipFile

def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    metadata = {
    
    "jobs": json_data["jobs"], #ile jobów wykonujemy
    "shots": json_data["shots"], #ile shotów dla każdego układu
    "repetitions": json_data["repetitions"], #ile shotów dla każdego układu
    "randomization": json_data["randomization"], #randomizacja kątów włączona/wyłączona
    "backend": json_data["backend"]
    }
    
    return metadata
#from JobResult import *

class BellResult:        
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table
        
    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = self.raw_results = pd.concat([self.raw_results,pd.DataFrame(result)], ignore_index=True)
        
    def Calculate(self,qubit):
        b=[]
        for i in range(8):
            a=[]
            selected_row = self.raw_results.loc[(self.raw_results["q"] == qubit) & (self.raw_results["i"] == i)]
            a.append(selected_row["000"].to_numpy()[0])
            a.append(selected_row["100"].to_numpy()[0])
            a.append(selected_row["010"].to_numpy()[0])
            a.append(selected_row["110"].to_numpy()[0])
            a.append(selected_row["001"].to_numpy()[0])
            a.append(selected_row["101"].to_numpy()[0])
            a.append(selected_row["011"].to_numpy()[0])
            a.append(selected_row["111"].to_numpy()[0])
            b.append(a)
            
        return b
        
    def SumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["i","q"], as_index = False).sum()

            
    
qq=13
#jn=4
jn=60
results_path = 'lg/results'

#metadata = load_json(results_path + '/data.json')
summed_result = BellResult(pd.DataFrame())
si=[[[] for _ in range(4)] for _ in range(qq)]
for i in range(jn): #metadata["jobs"]):
    pd_result = pd.read_csv(results_path + "/results_tests_" + str(i) + ".csv", index_col=0)
    
    summed_result.AppendResults(pd_result)
    rr = BellResult(pd.DataFrame())
    rr.AppendResults(pd_result)
    rr.SumResults()
    for d in range(qq):
        b=rr.Calculate(d)
        i=0
        s1=[]
        s2=[]
        n=0
        for j in range(8):
            n+=b[0][j]

summed_result.SumResults()
eps=0.1
for d in range(qq):
    b=summed_result.Calculate(d)
    n=0
    for j in range(8):
            n+=b[0][j]

    print("qubit:",d)
    print("Trials: ",n)
    ss=[]
    ac=[]
    ab=[]
    bc=[]
    aa=[]
    bb=[]
    print("xxC")
    for j in range(8):
        s=b[j][0]-b[j][1]-b[j][2]+b[j][3]-b[j][4]+b[j][5]+b[j][6]-b[j][7]
        sac=b[j][0]+b[j][1]-b[j][2]-b[j][3]-b[j][4]-b[j][5]+b[j][6]+b[j][7]
        sbc=b[j][0]-b[j][1]+b[j][2]-b[j][3]-b[j][4]+b[j][5]-b[j][6]+b[j][7]
        sab=b[j][0]-b[j][1]-b[j][2]+b[j][3]+b[j][4]-b[j][5]-b[j][6]+b[j][7]
        sa=b[j][0]+b[j][1]-b[j][2]-b[j][3]+b[j][4]+b[j][5]-b[j][6]-b[j][7]
        sb=b[j][0]-b[j][1]+b[j][2]-b[j][3]+b[j][4]-b[j][5]+b[j][6]-b[j][7]
        ss.append(s)
        ac.append(sac)
        ab.append(sab)
        bc.append(sbc)
        aa.append(sa)
        bb.append(sb)
        
        #print(*b[j])
        print(sum(b[j][:4])/n)
    print("ABC")
    print((ss[0]+ss[3]-ss[2]-ss[1])/(n*eps*eps*4))
    print("BAC")
    print((ss[4]+ss[7]-ss[6]-ss[5])/(n*eps*eps*4))
    print("AB")
    print((ab[0]+ab[3]-ab[2]-ab[1])/(n*eps*eps*4))
    print("BA")
    print((ab[4]+ab[7]-ab[6]-ab[5])/(n*eps*eps*4))
    print("AxC")
    print((ac[0]-ac[3]+ac[2]-ac[1])/(n*eps*4))
    print("xAC")
    print((ac[4]-ac[7]+ac[6]-ac[5])/(n*eps*4))
    print("xBC")
    print((bc[0]-bc[3]-bc[2]+bc[1])/(n*eps*4))
    print("BxC")
    print((bc[4]-bc[7]-bc[6]+bc[5])/(n*eps*4))
    print("Ax")
    print((aa[0]-aa[3]+aa[2]-aa[1])/(n*eps*4))
    print("xA")
    print((aa[4]-aa[7]+aa[6]-aa[5])/(n*eps*4))
    print("xB")
    print((bb[0]-bb[3]-bb[2]+bb[1])/(n*eps*4))
    print("Bx")
    print((bb[4]-bb[7]-bb[6]+bb[5])/(n*eps*4))
        
        




