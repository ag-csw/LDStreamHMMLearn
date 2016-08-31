from unittest import TestCase
import numpy as np
import pyemma.msm as MSM
import pyemma.msm.estimators as _MME
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from time import process_time
import matplotlib.pyplot as plt

from ldshmm.util.mm_family import MMFamily1

class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]

        self.taumeta = 3
        self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)

        self.nstep = 100 * self.taumeta
        self.nwindow = 10 * self.nstep
        self.numsteps = 100
        self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
        self.ntraj = 20
        self.r = (self.nwindow - self.nstep) / self.nwindow

        self.data1_0_0 = []
        for i in range(0, self.ntraj):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.lentraj))
        #print(self.data1_0_0)

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates = self.nstates)

        #import msmtools.estimation as msmest
        #Csparse = msmest.effective_count_matrix(dataslice, 2)
        #Csub = Csparse.toarray()
        #cshape = Csub.shape
        #C = np.zeros((self.nstates, self.nstates), dtype=float)
        #C[0:cshape[0], 0:cshape[1]] = Csub
        #print("Csparse:\n", Csparse)
        #print("Csub:\n", Csparse)
        #print("C:\n", C)
        return C


    def test_approach2(self):

        dataarray = np.asarray(self.data1_0_0)
        etime = np.zeros(self.numsteps + 2, dtype=float)
        err = np.zeros(self.numsteps + 1, dtype=float)
        etime[0] = 0
        for k in range(0, self.numsteps+1):
            data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
            dataslice0 = []
            for i in range(0, self.ntraj):
                dataslice0.append(data0[i, :])
            t0 = process_time()
            C0 = self.estimate_via_sliding_windows(dataslice0)  # count matrix for whole window
            t1 = process_time()
            A0 = _tm(C0)
            etime[k+1] = t1-t0 + etime[k]
            err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
        print("Times (Windows): ", etime)
        print("Errors (Windows): ", err)
        plot_result(etime, err, "naive", "numtraj=20, numsteps=100_8")


        print("\n############## Bayes #############")
        etimebayes = np.zeros(self.numsteps + 2, dtype=float)
        errbayes = np.zeros(self.numsteps + 1, dtype=float)

        weight0 = self.r
        weight1 = 1.0

        data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
        dataslice0 = []
        for i in range(0, self.ntraj):
            dataslice0.append(data0[i, :])

        t0 = process_time()
        C_old = self.estimate_via_sliding_windows(dataslice0)
        etimebayes[1] = process_time() - t0
        errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

        for k in range(1,self.numsteps+1):
            data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
            dataslice1new = []
            for i in range(0, self.ntraj):
                dataslice1new.append(data1new[i, :])
            t0 = process_time()
            C_new = self.estimate_via_sliding_windows(dataslice1new)  # count matrix for just new transitions

            C1bayes = weight0 * C_old + weight1 * C_new
            C_old = C1bayes
            t1 = process_time()
            etimebayes[k+1] = t1 - t0 + etimebayes[k]
            A1bayes = _tm(C1bayes)
            errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)
        print("Times (Bayes): ", etimebayes)
        print("Errors (Bayes): ", errbayes)
        plot_result(etimebayes, errbayes, "bayes", "numtraj=20, numsteps=100_8")
    

    def test_approach3(self):
        #generation of heatmap
        etime = np.zeros(self.numsteps + 2, dtype=float)
        err = np.zeros(self.numsteps + 1, dtype=float)
        etime[0] = 0
        avg_times = np.zeros((1,1))
        avg_errs = np.zeros((1,1))
        taumeta_values = [2] # 2,4,8    - 2 for testing if heatmap works in general
        eta_values = [50] # 50,100,200  - 50 for testing if heatmap works in general
        for one,taumeta in enumerate(taumeta_values):
            for two,eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables
                self.taumeta = taumeta
                self.nstep = 100 * self.taumeta
                self.nwindow = 10 * self.nstep
                self.numsteps = 100
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = 20
                self.r = (self.nwindow - self.nstep) / self.nwindow

                self.data1_0_0 = []
                for i in range(0, self.ntraj):
                    self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.lentraj))
                dataarray = np.asarray(self.data1_0_0)


                for k in range(0, self.numsteps + 1):
                    data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
                    dataslice0 = []
                    for i in range(0, self.ntraj):
                        dataslice0.append(data0[i, :])
                    t0 = process_time()
                    C0 = self.estimate_via_sliding_windows(dataslice0)  # count matrix for whole window
                    t1 = process_time()
                    A0 = _tm(C0)
                    etime[k + 1] = t1 - t0 + etime[k]
                    err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
                avg_time = sum(etime)/len(etime)
                avg_err = sum(err)/len(err)
                print("Avg Times (Windows): ",avg_time )
                print("Avg Errors (Windows): ", avg_err )

                print(one,two)
                avg_times[one][two]= avg_time
                avg_errs[one][two] = avg_err

        plot_result_heatmap(avg_times, taumeta_values, eta_values, "naive", "Basic Heatmap Performance")
        #plot_result_heatmap(avg_err, "naive", "Basic Heatmap Error")

        # the following lines are commented out to keep testing simple
"""
    print("\n############## Bayes #############")
    etimebayes = np.zeros(self.numsteps + 2, dtype=float)
    errbayes = np.zeros(self.numsteps + 1, dtype=float)

    weight0 = self.r
    weight1 = 1.0

    data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
    dataslice0 = []
    for i in range(0, self.ntraj):
        dataslice0.append(data0[i, :])

    t0 = process_time()
    C_old = self.estimate_via_sliding_windows(dataslice0)
    etimebayes[1] = process_time() - t0
    errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

    for k in range(1, self.numsteps + 1):
        data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
        dataslice1new = []
        for i in range(0, self.ntraj):
            dataslice1new.append(data1new[i, :])
        t0 = process_time()
        C_new = self.estimate_via_sliding_windows(dataslice1new)  # count matrix for just new transitions

        C1bayes = weight0 * C_old + weight1 * C_new
        C_old = C1bayes
        t1 = process_time()
        etimebayes[k + 1] = t1 - t0 + etimebayes[k]
        A1bayes = _tm(C1bayes)
        errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)
    print("Times (Bayes): ", etimebayes)
    print("Errors (Bayes): ", errbayes)
    plot_result_heatmap(etimebayes, errbayes, "bayes", "numtraj=20, numsteps=100_2")
"""
def plot_result_heatmap(data, x, y, type, heading):
    x, y = np.meshgrid(x, y)
    intensity = np.array(data)
    plt.pcolormesh(x, y, intensity)
    plt.colorbar()
    plt.title(heading)
    plt.savefig("test.png")

def plot_result(y_axis1_list, y_axis2_list, type, heading):
    """Plotting function for diagram with two y axes

    Parameters
        ----------
        y_axis1_list : list of elements for first y axis
        y_axis2_list : list of elements for second y axis
        type : string which characterizes the type of calculation (for instance "naive" or "bayes").
        heading : The custom heading for the plot title

        The two latter ones are just for plotting and saving the resulting plot. The filename will be type+ _ +heading
        """
    t_time = range(0, len(y_axis1_list))

    fig, ax1 = plt.subplots()
    ax1.plot(t_time, y_axis1_list, 'b-')
    ax1.set_xlabel('time t')
    ax1.set_ylabel('performance time')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()

    t_time = range(0, len(y_axis2_list))
    ax2.plot(t_time, y_axis2_list, 'r.')
    ax2.set_ylabel('error')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.title(heading)
    plt.savefig(type+'_'+heading+'.png')