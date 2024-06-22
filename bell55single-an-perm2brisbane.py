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
def matrix_minor2(arr, i, j,ii,jj):
    return np.linalg.det(np.delete(np.delete(arr,[i,ii],axis=0), [j,jj], axis=1)) 
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
    widet=0
    for j in range(1,len(P)):
        for k in range(1,len(P)):
            
            for jj in range(j):
                for kk in range(k):
                    aax=matrix_minor2(P,j,k,jj,kk)
                    #widet+=aax**2 *P[j][k] * (1 - P[j][k])*P[jj][kk] * (1 - P[jj][kk])
                    widet+=(2*((j+jj+k+kk)%2)-1)*aax *(P[j][k] * P[jj][kk]  - P[j][kk]*P[jj][k])
    standard_deviation-=staux*staux
            #print(standard_deviation)
    standard_deviation = standard_deviation/n
    return standard_deviation,widet/(n)
#from JobResult import *

class BellResult:        
    def __init__(self, results_table: pd.DataFrame) -> None:
        self.raw_results = results_table
        self.qq=1
        
    def AppendResults(self, result: pd.DataFrame) -> None:
        self.raw_results = pd.concat([self.raw_results,pd.DataFrame(result)], ignore_index=True)
        
    def Calculate(self):
        ind=[[""]]
        for q in range(12):
            inn=ind[-1]
            ind.append([])
            for s in inn:
                ind[-1].append("0"+s)
                ind[-1].append("1"+s)
        inn=ind[-1]
        b=0
        selected_row = self.raw_results
        for i in range(64*64):
            if inn[i] in selected_row.keys():
                b+=selected_row[inn[i]].sum()
        return b/self.qq
        
    def SumResults(self) -> None:
        self.raw_results = self.raw_results.sum()
    def CalculateMatrix(self,qubit,ina,inb,c) -> np.array:
        ind=["000","100","010","110","001","101","011","111"]
        ind=[[""]]
        for q in range(12):
            inn=ind[-1]
            ind.append([])
            for s in inn:
                ind[-1].append("0"+s)
                ind[-1].append("1"+s)
        inn=ind[-1]
        #print(len(inn))
        inx=[]
        for s in ina:
            for ss in inb:
                inx.append(s+ss)
        ind=["000","100","010","110","001","101","011","111"]
        ind=[[""]]
        for q in range(6):
            innh=ind[-1]
            ind.append([])
            for s in innh:
                ind[-1].append("0"+s)
                ind[-1].append("1"+s)
        iny=ind[-1]
        inz=[]
        for s in iny:
            for ss in inx:
                if c:
                    inz.append(s+ss)
                else:
                    inz.append(ss+s)
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
        su=0
        #selected_row = self.raw_results
        selected_row = self.raw_results
        #print(selected_row)
        for q in range(2**12):
            if inn[q] in selected_row.keys():
                su+=selected_row[inn[q]]
        for i in range(5):
            for j in range(5):
                for rr in range(2**6):
                    if inz[i+j*5+25*rr] in selected_row.keys():
                        self.matrix[i,j]+=selected_row[inz[i+j*5+25*rr]]
        divsu = lambda i: i / su
        vdivsu = np.vectorize(divsu)
        return vdivsu(self.matrix)
            
    
        
results_path = 'bris55_multi'
#results_path = 'belem-bell-dim21/wyniki'

#'nairobi_bell_sim/wyniki'

#metadata = load_json(results_path + '/data.json')
summed_result = BellResult(pd.DataFrame())
qq=1
sd=[0 for i in range(qq)]
ed=[0 for i in range(qq)]
ssd=[[] for i in range(qq)]
ii=0

for ih in range(20): #metadata["jobs"]):
    pd_result = pd.read_csv(results_path + "/wyniki_testy_" + str(ih) + ".csv", index_col=0)
    summed_result.AppendResults(pd_result)
    rr = BellResult(pd.DataFrame())
    rr.AppendResults(pd_result)
    rr.SumResults()
    ni=rr.Calculate()
    #for d in range(qq):
    #    for a in range(2):
    #        for b in range(2):
    #            P = rr.CalculateMatrix(a,b,d)
    #            dd=np.linalg.det(P)
    #            sd[d][a][b]+=dd
    #            ssd[d][a][b].append(dd)
    #            ed[d][a][b]+=std_dev(P,ni)
    ii+=1
summed_result.SumResults()
n=summed_result.Calculate()
print("Trials: ",n)

print("Dimension test")
ind=["000","001","010","011","100","101","110","111"]
inn=[]
for ff in range(2,8):
    inass=[ind[ff]]
    for gg in range(1,ff):
        inasb=inass.copy()
        inasb.append(ind[gg])
        for hh in range(gg):
            inasc=inasb.copy()
            inasc.append(ind[hh])
            inn.append(inasc)
innd=[]
for zz in range(len(inn)):
    innt=[]
    for xx in ind:
        if xx not in inn[zz]:
            innt.append(xx)
    innd.append(innt)
    #print(*innt)
qqq=1
cc=0
for d in range(qq):
    mx=0
    mm=0
    for ina in innd:
        for inb in innd:
        
                    P = summed_result.CalculateMatrix(d,ina,inb,cc)
                    sdev,sdaux=std_dev(P,n)
                    ssig=np.linalg.det(P)/np.sqrt(sdev)
                    if  ssig>mx or ssig<-mx:
                        if ssig>0:
                            mx=ssig
                        else:
                            mx=-ssig
                        sd[d]=0
                        ssd[d]=[]
                        ed[d]=0
                        for ih in range(20): #metadata["jobs"]):
                            pd_result = pd.read_csv(results_path + "/wyniki_testy_" + str(ih) + ".csv", index_col=0)
                            rr = BellResult(pd.DataFrame())
                            rr.AppendResults(pd_result)
                            rr.SumResults()
                            ni=rr.Calculate()
                            PP = rr.CalculateMatrix(d,ina,inb,cc)
                            dd=np.linalg.det(PP)
                            sd[d]+=dd
                            ssd[d].append(dd)
                            stdc,ssaux=std_dev(PP,ni)
                            ed[d]+=stdc
                        ssiga= sd[d]/np.sqrt(ed[d])
                        if ssiga>mm or ssiga<-mm:
                            if ssiga>0:
                                mm=ssiga
                            else:
                                mm=-ssiga
                            print("d: ",d)
                            print(*ina)
                            print(*inb)
                        
                            print('P =')
                            print(P)
                            print('Adj(P) =')
                            print(adj(P))
                            print('det(P) = ')
                            print(np.linalg.det(P), '±', np.sqrt(std_dev(P,n)[0]),'sig',np.linalg.det(P)/np.sqrt(std_dev(P,n)[0]))
                            print("avg: ",sd[d]/ii,'±',np.sqrt(ed[d])/ii,'sig',sd[d]/np.sqrt(ed[d]))
                            print("cor: ",sdaux)
                            for c in ssd[d]:
                                print(c)


