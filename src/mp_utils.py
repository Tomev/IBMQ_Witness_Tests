"""
A 

As suggested here:
https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
"""
from multiprocessing import Process, get_context
from multiprocessing.pool import Pool


class NonDaemonicProcess(Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NonDaemonicContext(type(get_context())):
    Process = NonDaemonicProcess


class NonDeamonicPool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NonDaemonicContext()
        super(NonDeamonicPool, self).__init__(*args, **kwargs)

