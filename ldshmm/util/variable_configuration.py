from ldshmm.util.variable_holder import Variable_Holder

class Variable_Config ():

    def __init__(self, iter_values1, iter_values2):
        self.taumeta = None
        self.eta = None
        self.scale_window = None
        self.num_trajectories = None
        self.statconc = Variable_Holder.mid_statconc
        self.timescaledisp = Variable_Holder.mid_timescaledisp
        self.iter_values1 = iter_values1
        self.iter_values2 = iter_values2
        self.heatmap_size = Variable_Holder.heatmap_size
        self.len_trajectory = Variable_Holder.len_trajectory_max

