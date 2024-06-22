import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
#from scipy.optimize import curve_fit
#from scipy.optimize import least_squares
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
            standard_deviation += (adj_P[j][k])**2 * P[k][j]
            staux+= adj_P[j][k] * P[k][j]
    standard_deviation-=staux*staux
            #print(standard_deviation)
    standard_deviation = standard_deviation/n
    return standard_deviation
#from JobResult import *

class BellResult:        
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table
        self.qq=7
        
    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = pd.concat([self.raw_results,pd.DataFrame(result)], ignore_index=True)
        
    def Calculate(self):
        ind=[[""]]
        for q in range(6):
            inn=ind[-1]
            ind.append([])
            for s in inn:
                ind[-1].append("0"+s)
                ind[-1].append("1"+s)
        inn=ind[-1]
        b=0
        selected_row = self.raw_results
        for i in range(64):
            if inn[i] in selected_row.keys():
                b+=selected_row[inn[i]].sum()
        return b/self.qq
        
    def SumResults(self) -> None:
        self.raw_results = self.raw_results.groupby(["q"], as_index = False).sum()
    def CalculateMatrix(self,a,b,qubit) -> np.array:
        ind1=["000","100","010","110","001","101","011","111"]
        ind=["000","100","010","110","001","101","011","111"]
        inn=[]
        for s in ind:
            for ss in ind1:
                inn.append(s+ss)
        #inn=["" for i in range(64)]
        #for i in range(64):
        #    ia=i%8
        #    ib=i//8
        #    iaa=ia%2
        #    iaaa=(2*ia+iaa)%8
        #    ibb=ib%2
        #    ibbb=(2*ib+ibb)%8
        #    inn[ibbb*8+iaaa]=inn2[i]
        
        #print(*inn)
        self.matrix = np.zeros([5,5])
        self.matrix[4,4]=1
        su=0
        #selected_row = self.raw_results
        selected_row = self.raw_results.loc[(self.raw_results["q"] == qubit)]
        for q in range(2**6):
            if inn[q] in selected_row.keys():
                su+=selected_row[inn[q]].iloc[0]
        for i in range(4):
            for j in range(4):
                if inn[i+j*8+4*a+32*b] in selected_row.keys():
                    self.matrix[i,j]=selected_row[inn[i+j*8+4*a+32*b]].iloc[0]
            self.matrix[i,4]=0
            for q in range(4):
                if inn[i+32*(1-b)+q*8+4*a] in selected_row.keys():
                    self.matrix[i,4]+=selected_row[inn[i+32*(1-b)+q*8+4*a]].iloc[0]
        for j in range(4):
            self.matrix[4,j]=0
            for q in range(4):
                if inn[q+4*(1-a)+j*8+32*b] in selected_row.keys():
                    self.matrix[4,j]+=selected_row[inn[q+4*(1-a)+j*8+32*b]].iloc[0]
        self.matrix[4,4]=0
        for i in range(4):
            for j in range(4):
                if inn[i+4*(1-a)+32*(1-b)+j*8] in selected_row.keys():
                    self.matrix[4,4]+=selected_row[inn[i+4*(1-a)+32*(1-b)+j*8]].iloc[0]
        divsu = lambda i: i / su
        vdivsu = np.vectorize(divsu)
        return vdivsu(self.matrix)
            
    
        
results_path = 'bell55multi/results'
#results_path = 'belem-bell-dim21/wyniki'

#'nairobi_bell_sim/wyniki'

#metadata = load_json(results_path + '/data.json')
summed_result = BellResult(pd.DataFrame())
qq=7
sd=[[[0,0],[0,0]] for i in range(qq)]
ed=[[[0,0],[0,0]] for i in range(qq)]
ssd=[[[[],[]],[[],[]]] for i in range(qq)]
ii=0

for ih in range(30): #metadata["jobs"]):
    pd_result = pd.read_csv(results_path + "/results_tests_" + str(ih) + ".csv", index_col=0)
    summed_result.AppendResults(pd_result)
    rr = BellResult(pd.DataFrame())
    rr.AppendResults(pd_result)
    rr.SumResults()
    ni=rr.Calculate()
    for d in range(qq):
        for a in range(2):
            for b in range(2):
                P = rr.CalculateMatrix(a,b,d)
                dd=np.linalg.det(P)
                sd[d][a][b]+=dd
                ssd[d][a][b].append(dd)
                ed[d][a][b]+=std_dev(P,ni)
    ii+=1
summed_result.SumResults()
n=summed_result.Calculate()
print("Trials: ",n)

print("Dimension test")
for d in range(qq):
    for a in range(2):
        for b in range(2):
            print("dab: ",d,a,b)
            P = summed_result.CalculateMatrix(a,b,d)
            print('P =')
            print(P)
            print('Adj(P) =')
            print(adj(P))
            print('det(P) = ')
            print(np.linalg.det(P), '±', np.sqrt(std_dev(P,n)),'sig',np.linalg.det(P)/np.sqrt(std_dev(P,n)))
            print("avg: ",sd[d][a][b]/ii,'±',np.sqrt(ed[d][a][b])/ii,'sig',sd[d][a][b]/np.sqrt(ed[d][a][b]))
            for c in ssd[d][a][b]:
                print(c)



