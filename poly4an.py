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
        for i in range(16):
            a=[0 for j in range(16)]
            selected_row = self.raw_results.loc[(self.raw_results["q"] == qubit) & (self.raw_results["i"] == i)]
            for ke,va in selected_row.items():
                if len(ke)==4:
                    z=0
                    j=1
                    for c in ke[::-1]:
                        if c=='1':
                            z+=j
                        j<<=1
                    
                        
                    a[z]=va.to_numpy()[0]
            b.append(a)
        return b
        
    def SumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["i","q"], as_index = False).sum()

            
    
qq=10
#jn=4
jn=2
results_path = 'poly/results'
summed_result = BellResult(pd.DataFrame())
for i in range(jn):
    pd_result = pd.read_csv(results_path + "/results_tests_" + str(i) + ".csv", index_col=0)
    #pd_result = pd.read_csv(results_path + "/ibm_torino_(0, 1, 2, 3, 4)" + ".csv", index_col=0)
    summed_result.AppendResults(pd_result)
summed_result.SumResults()
maa=0
for d in range(qq):
    
    b=summed_result.Calculate(d)
    i=0
    s1=[]
    s2=[]
    n=2*sum(b[0])
    print("Set",d)
    print("Trials",n//2)
    #minimal M
    MM=10
    for j in range(4):
        #xxxx
        ee=[0 for w in range(8)]
        sig=0
        e=0
        pj=1<<j
        aa=[[0 for k in range(8)] for m in range(8)]
        for k in range(16):
            for m in range(16):
                aa[m%pj+((m//pj)//2)*pj][k%pj+((k//pj)//2)*pj]+=b[m][k]
        pc=[1,1,1,1,-1,-1,-1,-1]
        pb=[1,1,-1,-1,1,1,-1,-1]
        pa=[1,-1,1,-1,1,-1,1,-1]
        ae=0
        ff=[0 for w in range(8)]
        fi=[0 for w in range(8)]
        si=[0 for w in range(8)]
        for w in range(8):
            
            e+=aa[0][w]*((pa[w]+pb[w]+pc[w])/4-pa[w]*pb[w]*pc[w])
            
            ee[0]+=aa[0][w]*((pa[w]+pb[w]+pc[w])/4-pa[w]*pb[w]*pc[w])
            si[0]+=aa[0][w]*((pa[w]+pb[w]+pc[w])/4-pa[w]*pb[w]*pc[w])**2
            e+=aa[1][w]*((pb[w]+pc[w])/4+pa[w]*(pb[w]+pc[w])/2-pa[w]*pb[w]*pc[w])
            ee[1]+=aa[1][w]*((pb[w]+pc[w])/4+pa[w]*(pb[w]+pc[w])/2-pa[w]*pb[w]*pc[w])
            si[1]+=aa[1][w]*((pb[w]+pc[w])/4+pa[w]*(pb[w]+pc[w])/2-pa[w]*pb[w]*pc[w])**2
            e+=aa[2][w]*((pa[w]+pc[w])/4+pb[w]*(pa[w]+pc[w])/2-pa[w]*pb[w]*pc[w])
            ee[2]+=aa[2][w]*((pa[w]+pc[w])/4+pb[w]*(pa[w]+pc[w])/2-pa[w]*pb[w]*pc[w])
            si[2]+=aa[2][w]*((pa[w]+pc[w])/4+pb[w]*(pa[w]+pc[w])/2-pa[w]*pb[w]*pc[w])**2
            
            e+=aa[4][w]*((pa[w]+pb[w])/4+pc[w]*(pa[w]+pb[w])/2-pa[w]*pb[w]*pc[w])
            ee[4]+=aa[4][w]*((pa[w]+pb[w])/4+pc[w]*(pa[w]+pb[w])/2-pa[w]*pb[w]*pc[w])
            si[4]+=aa[4][w]*((pa[w]+pb[w])/4+pc[w]*(pa[w]+pb[w])/2-pa[w]*pb[w]*pc[w])**2
            e+=aa[3][w]*(-pa[w]*pb[w]/2+pc[w]/4+pc[w]*(pa[w]+pb[w])/2)
            ee[3]+=aa[3][w]*(-pa[w]*pb[w]/2+pc[w]/4+pc[w]*(pa[w]+pb[w])/2)
            si[3]+=aa[3][w]*(-pa[w]*pb[w]/2+pc[w]/4+pc[w]*(pa[w]+pb[w])/2)**2
            e+=aa[5][w]*(-pc[w]*pa[w]/2+pb[w]/4+pb[w]*(pc[w]+pa[w])/2)
            ee[5]+=aa[5][w]*(-pc[w]*pa[w]/2+pb[w]/4+pb[w]*(pc[w]+pa[w])/2)
            si[5]+=aa[5][w]*(-pc[w]*pa[w]/2+pb[w]/4+pb[w]*(pc[w]+pa[w])/2)**2
            e+=aa[6][w]*(-pc[w]*pb[w]/2+pa[w]/4+pa[w]*(pb[w]+pc[w])/2)
            ee[6]+=aa[6][w]*(-pc[w]*pb[w]/2+pa[w]/4+pa[w]*(pb[w]+pc[w])/2)
            si[6]+=aa[6][w]*(-pc[w]*pb[w]/2+pa[w]/4+pa[w]*(pb[w]+pc[w])/2)**2
            e+=aa[7][w]*(pa[w]*pb[w]*pc[w]-(pa[w]*pb[w]+pb[w]*pc[w]+pc[w]*pa[w])/2)
            ee[7]+=aa[7][w]*(pa[w]*pb[w]*pc[w]-(pa[w]*pb[w]+pb[w]*pc[w]+pc[w]*pa[w])/2)
            si[7]+=aa[7][w]*(pa[w]*pb[w]*pc[w]-(pa[w]*pb[w]+pb[w]*pc[w]+pc[w]*pa[w])/2)**2
        print(si[0],si[7])
        for w in range(8):
            ee[w]*=ee[w]
        sig=sum(si)-sum(ee)/n
        
        print("Sliwa inequality class 5 correlations excl", j)
        print("S = ",e/n,"+-",np.sqrt(sig)/n) 
        MM=min(e/n,MM)
    print("Minimal violation:",MM)

        
        






