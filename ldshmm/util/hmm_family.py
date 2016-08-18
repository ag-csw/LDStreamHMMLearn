
class HMM_Family(object):


    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")

class HMM_Family1(HMM_Family):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State
    # Crisply-clustered observables
    # Initial Distribution is the Stationary Distributino

    def __init__(self, nobserved, clusterconc, withinclusterconce):
        self.nobserved = nobserved
        self.clusterconc = clusterconc
        self.withinclusterconc = withinclusterconce

    def sample(self,size = 1):
        raise NotImplementedError("Please implement this method")