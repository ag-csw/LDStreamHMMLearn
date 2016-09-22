import matplotlib.pyplot as plt
import numpy as np



class ComplexPlot():
    cols = 2
    current = 1
    im = None

    def new_plot(self, heading, rows):
        plt.figure()
        plt.suptitle(heading)
        self.rows=rows+1


    def add_to_plot_separate_colorbar(self, data_naive, data_bayes, x_labels, y_labels, y_label):
        plt.subplot(self.rows,self.cols,self.current)
        plt.pcolor(data_naive, cmap="Reds")
        plt.xticks(np.arange(len(x_labels)), (str(x_label) for x_label in x_labels))
        plt.yticks(np.arange(len(y_labels)), (str(y_label) for y_label in y_labels))
        plt.xlabel("taumeta")
        plt.ylabel(y_label)
        plt.colorbar()
        plt.tight_layout(2)
        self.current = self.current+1

        plt.subplot(self.rows,self.cols,self.current)
        plt.pcolor(data_bayes, cmap="Reds")
        plt.xticks(np.arange(len(x_labels)), (str(x_label) for x_label in x_labels))
        plt.yticks(np.arange(len(y_labels)), (str(y_label) for y_label in y_labels))
        plt.xlabel("taumeta")
        plt.ylabel(y_label)
        plt.colorbar()
        plt.tight_layout(2)
        self.current = self.current + 1

    def add_to_plot_same_colorbar (self, data_naive, data_bayes, x_labels, y_labels, y_label, minimum, maximum):
        plt.subplot(self.rows, self.cols, self.current)
        self.im = plt.pcolor(data_naive, vmin=minimum, vmax=maximum, cmap="Reds")
        plt.xticks(np.arange(len(x_labels)), (str(x_label) for x_label in x_labels))
        plt.yticks(np.arange(len(x_labels)), (str(y_label) for y_label in y_labels))
        plt.xlabel("taumeta")
        plt.ylabel(y_label)
        plt.tight_layout(2)
        self.current = self.current+1

        plt.subplot(self.rows, self.cols, self.current)
        self.im = plt.pcolor(data_bayes, vmin=minimum, vmax=maximum, cmap="Reds")
        plt.xticks(np.arange(len(x_labels)), (str(x_label) for x_label in x_labels))
        plt.yticks(np.arange(len(y_labels)), (str(y_label) for y_label in y_labels))
        plt.xlabel("taumeta")
        plt.ylabel(y_label)
        plt.tight_layout(2)
        self.current = self.current + 1

    def save_plot_same_colorbar(self, heading):
        ax = plt.subplot(self.rows, self.cols, self.rows*self.cols)
        cbar = plt.colorbar(self.im, ax=ax, orientation="horizontal")
        labels = cbar.ax.get_xticklabels()
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.set_xticklabels(labels=labels, rotation=45)
        plt.delaxes(ax)

        plt.savefig(heading + ".png")


    def save_plot_separate_colorbars(self, heading):
        plt.savefig(heading + ".png")

class LinePlot():

    cols = 2
    current = 1

    def new_plot(self, heading, rows):
        plt.figure()
        plt.suptitle(heading)
        self.rows=rows

    def add_to_plot(self, data_naive, data_bayes, x_labels, y_label, evaluated_param_name):
        plt.subplot(self.rows, self.cols, self.current)
        plt.xlabel('taumeta')
        plt.ylabel(y_label)

        for bayes_array_piece in data_naive:
            x_axis = range(0, len(bayes_array_piece))
            plt.plot(x_axis, bayes_array_piece)

        plt.legend([evaluated_param_name + " = " + str(x) for x in x_labels], loc='upper center', ncol=3,
                   fancybox=True, fontsize=6)
        plt.tight_layout(2)
        self.current = self.current + 1


        plt.subplot(self.rows, self.cols, self.current)
        plt.xlabel('taumeta')
        plt.ylabel(y_label)

        for bayes_array_piece in data_bayes:
            x_axis = range(0, len(bayes_array_piece))
            plt.plot(x_axis, bayes_array_piece)

        plt.legend([evaluated_param_name + " = " + str(x) for x in x_labels], loc='upper center', ncol=3,
          fancybox=True, fontsize=6)
        plt.tight_layout(2)
        self.current = self.current + 1

    def save_plot(self, heading):
        plt.savefig(heading + ".png")



def plot_result_heatmap(data_naive, data_bayes, x_labels, y_labels, y_axis_name, type, heading):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolor(data_naive, cmap="Reds")
    plt.title(heading)
    plt.xticks(np.arange(3), (str(x_label) for x_label in x_labels))
    plt.yticks(np.arange(3), (str(y_label) for y_label in y_labels))
    plt.xlabel("taumeta")
    plt.ylabel(y_axis_name)
    plt.title("Naive " + type)
    plt.colorbar()
    plt.tight_layout(2)

    plt.subplot(1, 2, 2)
    plt.pcolor(data_bayes, cmap="Reds")
    plt.title("Bayes " + type)
    plt.xticks(np.arange(3), (str(x_label) for x_label in x_labels))
    plt.yticks(np.arange(3), (str(y_label) for y_label in y_labels))
    plt.xlabel("taumeta")
    plt.ylabel(y_axis_name)
    plt.colorbar()
    plt.tight_layout(2)

    plt.savefig(heading + ".png")

def plot_result(y_axis1_list, y_axis2_list, type, heading):
    """Plotting function for diagram with two y axes

    Parameters
        ----------
        y_axis1_list : list of elements for first y axis
        y_axis2_list : list of elements for second y axis
        type : string which characterizes the type subplotsof calculation (for instance "naive" or "bayes").
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
    plt.savefig(type + '_' + heading + '.png')

line = LinePlot()
line.new_plot("Test", 2)
data = [[1,2,3],[2,3,4],[8,4,7]]
x_labels = [2,3,4]
line.add_to_plot(data_naive=data, data_bayes=data, y_label="Performance Slope", x_labels=x_labels,evaluated_param_name="eta")
line.add_to_plot(data_naive=data, data_bayes=data, y_label="Performance Slope", x_labels=x_labels, evaluated_param_name="eta")
line.save_plot("Lineplot")