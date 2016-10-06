from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1


class MM_Stationarity():

    def __init__(self, delta=0):
        self.nstates = 4

        if delta==0:
            self.mmf1_0 = MMFamily1(self.nstates)
            self.sample = self.mmf1_0.sample()[0]
        else:
            self.timescaledisp = 2.0
            self.statconc = 0.05
            self.mmf1_0 = MMFamily1(self.nstates, self.timescaledisp, self.statconc)
            self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=delta)
            self.sample = self.qmmf1_0.sample()[0]