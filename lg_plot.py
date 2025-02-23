import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
plt.rcParams['text.usetex'] = True
plt.rcParams["legend.loc"] = 'lower right' 

rp=["lgy/results-br1/results-br1","lgy/results-br/results-br",\
    "lgy/results-sh1/results-sh1","lgy/results-sh/results-sh",\
    "lgy/results-ky1/results-ky1","lgy/results-ky/results-ky"]
iv_LGBA=[]
ie_LGBA=[]
iv_LGAB=[]
ie_LGAB=[]
iv_AbC=[]
iv_bAC=[]
ie_AbC=[]
ie_bAC=[]
iv_aBC=[]
iv_BaC=[]
ie_aBC=[]
ie_BaC=[]
iv_ABC=[]
iv_BAC=[]
ie_ABC=[]
ie_BAC=[]
iv_Ab=[]
iv_bA=[]
ie_Ab=[]
ie_bA=[]
iv_aB=[]
iv_Ba=[]
ie_aB=[]
ie_Ba=[]
iv_abC=[]
iv_baC=[]
ie_abC=[]
ie_baC=[]
iv_AB=[]
iv_BA=[]
ie_AB=[]
ie_BA=[]
orv=[]
ore=[]
for results_path in rp: 
        pdr = pd.read_csv(
            results_path + ".csv", index_col=0
        )
        iv_LGBA.append(pdr.loc[:,"LGBA"])
        ie_LGBA.append(pdr.loc[:,"LGeBA"])
        iv_LGAB.append(pdr.loc[:,"LGAB"])
        ie_LGAB.append(pdr.loc[:,"LGeAB"])
        iv_AbC.append(pdr.loc[:,"AbC"])
        ie_AbC.append(pdr.loc[:,"eAbC"])
        iv_bAC.append(pdr.loc[:,"bAC"])
        ie_bAC.append(pdr.loc[:,"ebAC"])
        iv_aBC.append(pdr.loc[:,"aBC"])
        ie_aBC.append(pdr.loc[:,"eaBC"])
        iv_BaC.append(pdr.loc[:,"BaC"])
        ie_BaC.append(pdr.loc[:,"eBaC"])
        iv_ABC.append(pdr.loc[:,"ABC"])
        ie_ABC.append(pdr.loc[:,"eABC"])
        iv_BAC.append(pdr.loc[:,"BAC"])
        ie_BAC.append(pdr.loc[:,"eBAC"])
        iv_Ab.append(pdr.loc[:,"Ab"])
        ie_Ab.append(pdr.loc[:,"eAb"])
        iv_bA.append(pdr.loc[:,"bA"])
        ie_bA.append(pdr.loc[:,"ebA"])
        iv_aB.append(pdr.loc[:,"aB"])
        ie_aB.append(pdr.loc[:,"eaB"])
        iv_Ba.append(pdr.loc[:,"Ba"])
        ie_Ba.append(pdr.loc[:,"eBa"])
        iv_abC.append(pdr.loc[:,"abC"])
        ie_abC.append(pdr.loc[:,"eabC"])
        iv_baC.append(pdr.loc[:,"baC"])
        ie_baC.append(pdr.loc[:,"ebaC"])
        iv_AB.append(pdr.loc[:,"AB"])
        ie_AB.append(pdr.loc[:,"eAB"])
        iv_BA.append(pdr.loc[:,"BA"])
        ie_BA.append(pdr.loc[:,"eBA"])
        orv.append(pdr.loc[:,"order"])
        ore.append(pdr.loc[:,"eorder"])
        #print(*inequality_values_BA)
        #print(*inequality_errors_BA)
ddb=[0,1,2,3,4,5,6,7,8,9]
figp=["br","sh","ky"]
naa=["brisbane","sherbrooke","kyiv"]
plt.rcParams["legend.loc"] = 'lower right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3):
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)

    ax[q].add_patch( Rectangle((-0.5, 0), 
                        10, 1, 
                        ec ='none',  
                        fc ='yellow', 
                        lw = 0,label="_nolegend_" )) 
    ax[q].axhline(1.4142,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_LGAB[2*q], ie_LGAB[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_LGAB[2*q+1], ie_LGAB[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_LGBA[2*q], ie_LGBA[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_LGBA[2*q+1], ie_LGBA[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: AB$",r"$2ECR\: AB$",r"$1ECR\: BA$",r"$2ECR\: BA$"])
    ax[q].set_ylim([0,1.5])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .5), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/lg.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3):      
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)

    #ax.add_patch( Rectangle((-0.5, 0), 
    #                    10, 1, 
    #                    ec ='none',  
    #                    fc ='yellow', 
    #                    lw = 0,label="_nolegend_" )) 
    ax[q].axhline(1,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,orv[2*q], ore[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,orv[2*q+1], ore[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].legend([r"$1ECR$",r"$2ECR$"])
    ax[q].set_ylim([0,1.5])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .8), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/order.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()
plt.rcParams["legend.loc"] = 'upper right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(-0.7071,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_AbC[2*q], ie_AbC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_AbC[2*q+1], ie_AbC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_bAC[2*q], ie_bAC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_bAC[2*q+1], ie_bAC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: AbC$",r"$2ECR\: AbC$",r"$1ECR\: bAC$",r"$2ECR\: bAC$"])
    ax[q].set_ylim([-1,0])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .2), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/AC.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()

plt.rcParams["legend.loc"] = 'lower right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0.7071,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_aBC[2*q], ie_aBC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_aBC[2*q+1], ie_aBC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_BaC[2*q], ie_BaC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_BaC[2*q+1], ie_BaC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: aBC$",r"$2ECR\: aBC$",r"$1ECR\: BaC$",r"$2ECR\: BaC$"])
    ax[q].set_ylim([0,1])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .8), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/BC.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()
    
plt.rcParams["legend.loc"] = 'center left' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0.5,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].axhline(-0.5,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_ABC[2*q], ie_ABC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_ABC[2*q+1], ie_ABC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_BAC[2*q], ie_BAC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_BAC[2*q+1], ie_BAC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: ABC$",r"$2ECR\:ABC$",r"$1ECR\: BAC$",r"$2ECR\: BAC$"])
    ax[q].set_ylim([-0.7,0.7])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .5), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(-10,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/ABC.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()

plt.rcParams["legend.loc"] = 'lower right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0.7071,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_Ab[2*q], ie_Ab[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_Ab[2*q+1], ie_Ab[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_bA[2*q], ie_bA[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_bA[2*q+1], ie_bA[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: Ab$",r"$2ECR\: Ab$",r"$1ECR\: bA$",r"$2ECR\: bA$"])
    ax[q].set_ylim([0,1])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .8), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/A.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()

plt.rcParams["legend.loc"] = 'lower right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0.7071,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_aB[2*q], ie_aB[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_aB[2*q+1], ie_aB[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_Ba[2*q], ie_Ba[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_Ba[2*q+1], ie_Ba[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: aB$",r"$2ECR\: aB$",r"$1ECR\: Ba$",r"$2ECR\: Ba$"])
    ax[q].set_ylim([0,1])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .8), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/B.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()

plt.rcParams["legend.loc"] = 'upper right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_abC[2*q], ie_abC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_abC[2*q+1], ie_abC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_baC[2*q], ie_baC[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_baC[2*q+1], ie_baC[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: abC$",r"$2ECR\: abC$",r"$1ECR\: baC$",r"$2ECR\: baC$"])
    ax[q].set_ylim([-0.5,0.5])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .7), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(-10,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
plt.savefig(r'lgy/C.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
plt.close()

plt.rcParams["legend.loc"] = 'upper right' 
fig,ax=plt.subplots(3,1, figsize=(5, 9),tight_layout=True)
for q in range(3): 
    if q<2:
        ax[q].set_xticks(ddb,[])
    else:
        ax[q].set_xticks(ddb)
    ax[q].axhline(0,  color="black",linewidth=0.5, label="_nolegend_")
    ax[q].errorbar(ddb,iv_AB[2*q], ie_AB[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='blue')
    ax[q].errorbar(ddb,iv_AB[2*q+1], ie_AB[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='darkgreen')
    ax[q].errorbar(ddb,iv_BA[2*q], ie_BA[2*q], linewidth=0, capsize=3, elinewidth=1, marker=".", color='red')
    ax[q].errorbar(ddb,iv_BA[2*q+1], ie_BA[2*q+1], linewidth=0, capsize=3, elinewidth=1, marker=".", color='orange')
    ax[q].legend([r"$1ECR\: AB$",r"$2ECR\: AB$",r"$1ECR\: BA$",r"$2ECR\: BA$"])
    ax[q].set_ylim([-0.5,0.5])
    ax[q].set_xlim([-0.5,9.5])
    ax[q].annotate(f'{naa[q]}',xy=(0.5, .7), xycoords='axes fraction')
    ax[q].ticklabel_format(axis='y', style='sci', scilimits=(-10,0))
    ax[q].get_yaxis().get_offset_text().set_visible(False)
    plt.savefig(r'lgy/AB.pdf', bbox_inches='tight',pad_inches=0,dpi=300)
