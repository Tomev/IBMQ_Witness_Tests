"""
This module contains settings for the project.
"""

from os import environ,remove
from zipfile import ZipFile

# IBMQ token environmental variables
# naming convention: IBMQ_Token_<TOKEN_ID>
# "IBMQ_Token_KM","IBMQ_Token_MS",
TOKEN_VARIABLES = ['IBMQ_Token_A01','IBMQ_Token_A02','IBMQ_Token_A03','IBMQ_Token_A04','IBMQ_Token_A05',\
                   'IBMQ_Token_A06','IBMQ_Token_A07','IBMQ_Token_A08','IBMQ_Token_A09','IBMQ_Token_A10',\
                   'IBMQ_Token_A11','IBMQ_Token_A12','IBMQ_Token_A13','IBMQ_Token_A14','IBMQ_Token_A15']
TOKEN_VARIABLESG =['IBMQ_Token_ARO'] #,'IBMQ_Token_AZA']
#,'IBMQ_Token_NGL']
                  #,'IBMQ_Token_RZA',\
TOKEN_VARIABLESX = ['IBMQ_Token_EKO','IBMQ_Token_DEF','IBMQ_Token_AMZ','IBMQ_Token_MOG','IBMQ_Token_YAR',\
                   'IBMQ_Token_BEF','IBMQ_Token_CUD','IBMQ_Token_FAP','IBMQ_Token_HTF','IBMQ_Token_NAR',\
                   'IBMQ_Token_401','IBMQ_Token_BS2','IBMQ_Token_SEL','IBMQ_Token_FRZ','IBMQ_Token_PKO']
TOKEN_VARIABLESY  =['IBMQ_Token_ARO','IBMQ_Token_ANR','IBMQ_Token_HZA','IBMQ_Token_SRO','IBMQ_Token_IZA']
TOKEN_VARIABLESF= ['IBMQ_Token_YQ2','IBMQ_Token_N9Y','IBMQ_Token_MOC','IBMQ_Token_PBA','IBMQ_Token_E94']
#'IBMQ_Token_AN','IBMQ_Token_ISA'
TOKEN_VARIABLESE = ['IBMQ_Token_ANT''IBMQ_Token_PAU','IBMQ_Token_KAR','IBMQ_Token_PAU','IBMQ_Token_KAR']
TOKEN_VARIABLESD = ['IBMQ_Token_ASA','IBMQ_Token_RSL','IBMQ_Token_GKA','IBMQ_Token_GOK','IBMQ_Token_MKR',\
                   'IBMQ_Token_AKR','IBMQ_Token_OKR','IBMQ_Token_FJA','IBMQ_Token_AJA','IBMQ_Token_JPA',\
                   'IBMQ_Token_MNA','IBMQ_Token_FKR','IBMQ_Token_ODZ','IBMQ_Token_JDA','IBMQ_Token_HDA',\
                   'IBMQ_Token_MBA','IBMQ_Token_ABA','IBMQ_Token_ANC','IBMQ_Token_ADA','IBMQ_Token_ISS',\
                   'IBMQ_Token_BST','IBMQ_Token_IBA','IBMQ_Token_SSL','IBMQ_Token_RBE','IBMQ_Token_DKR']
TOKEN_VARIABLESC = ['IBMQ_Token_SJA','IBMQ_Token_KJA','IBMQ_Token_IR','IBMQ_Token_PAS','IBMQ_Token_KST',\
                   'IBMQ_Token_MTO','IBMQ_Token_RPU','IBMQ_Token_ECZ','IBMQ_Token_BGL','IBMQ_Token_IGL',\
                   'IBMQ_Token_PST','IBMQ_Token_AST','IBMQ_Token_JMU','IBMQ_Token_MBI','IBMQ_Token_ABB',\
                   'IBMQ_Token_ABU','IBMQ_Token_EBB','IBMQ_Token_HSA','IBMQ_Token_KSA','IBMQ_Token_SGG',\
                   'IBMQ_Token_DS','IBMQ_Token_JST','IBMQ_Token_FST','IBMQ_Token_IST','IBMQ_Token_JGL']
TOKEN_VARIABLESB = ['IBMQ_Token_MN','IBMQ_Token_PS','IBMQ_Token_SS','IBMQ_Token_AR',\
                   'IBMQ_Token_AK','IBMQ_Token_MD','IBMQ_Token_GD','IBMQ_Token_SM','IBMQ_Token_SW',\
                   'IBMQ_Token_RK','IBMQ_Token_KK','IBMQ_Token_WS','IBMQ_Token_NS','IBMQ_Token_ZS',\
                   'IBMQ_Token_RS','IBMQ_Token_JBA','IBMQ_Token_JJB','IBMQ_Token_KGO','IBMQ_Token_JGO',\
                   'IBMQ_Token_WEW','IBMQ_Token_WAW','IBMQ_Token_RKR','IBMQ_Token_MZA','IBMQ_Token_EBA',\
                   'IBMQ_Token_HTE','IBMQ_Token_STE','IBMQ_Token_FDA']

TOKEN_VARIABLESA = ["IBMQ_Token_AB","IBMQ_Token_KM","IBMQ_Token_MS","IBMQ_Token_EB","IBMQ_Token_TR",\
                   "IBMQ_Token_TB","IBMQ_Token_JT","IBMQ_Token_JB","IBMQ_Token_HG","IBMQ_Token_KG",\
                   "IBMQ_Token_ZB","IBMQ_Token_MB","IBMQ_Token_SG","IBMQ_Token_IG","IBMQ_Token_WB",\
                   "IBMQ_Token_KB","IBMQ_Token_PB","IBMQ_Token_IB","IBMQ_Token_BB","IBMQ_Token_BZ",\
                   "IBMQ_Token_TM","IBMQ_Token_PJ","IBMQ_Token_MK","IBMQ_Token_LM","IBMQ_Token_BC"]
TOKENS = {key: environ[key] for key in TOKEN_VARIABLES}
TOKENSA = {key: environ[key] for key in TOKEN_VARIABLESA}

N_TEST_CIRCUITS = 0  # Number of test circuits at the beginning of the job.
N_REPETITIONS = 100  # Number of experiments repetitions.
N_JOBS = 30  # Number of jobs we want to submit.
N_SHOTS = 10000
WAIT_TIME = 30  # Delay (in seconds) between checking job status.
SHOULD_RANDOMIZE = True  # Circuits order randomization.
AUTHOR_INITIALS = "TB"
DESCRIPTION = "Short description of a job"

RESULTS_FOLDER_NAME = "results"
ZIP_FILE_NAME = "results-55"

def experiments_cleen_up(job_list_path: str) -> None:
    with ZipFile(ZIP_FILE_NAME + ".zip", "a") as zip_file:
        zip_file.write(job_list_path, arcname="results/job_list.csv")

    try:
        remove(job_list_path)
    except Exception as alert:
        print(alert)
