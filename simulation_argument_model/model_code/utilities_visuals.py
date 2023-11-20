# import of necessary libraries
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import matplotlib as mpl


def transform_SyPaAn_single_measure_single_dependency(SyPaAn_data, measure, depending_variables):

    result_array = np.zeros(len(depending_variables)+1)

    for single_param_comb in SyPaAn_data:

        single_data_point = np.zeros((len(depending_variables)+1, len(single_param_comb[measure])))
        single_data_point[0, :] = np.array(single_param_comb[measure])

        for index, dependency in enumerate(depending_variables):
            single_data_point[1+index, :] = single_param_comb[dependency]

        result_array = np.c_[result_array, single_data_point]

    return result_array[:, 1:]


def xy_plot_measurement_boxplot(x_axis, y_axis, SyPaAn_data):
    data_points = transform_SyPaAn_single_measure_single_dependency(SyPaAn_data, y_axis, [x_axis])

    x_values = np.unique(data_points[1])
    y_values = data_points[0].reshape(len(x_values),int(len(data_points[0])/len(x_values))).transpose()

    fig = plt.figure(figsize=(15,9))

    plt.boxplot(y_values, positions=np.round(x_values,2), widths=0.075)
    plt.title(f"Plot between {x_axis} and {y_axis}")
    plt.xticks(rotation=90)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def xy_plot_measurement(x_axis, y_axis, SyPaAn_data, log_scale=False):
    data_points = transform_SyPaAn_single_measure_single_dependency(SyPaAn_data, y_axis, [x_axis])

    fig = plt.figure()

    if log_scale:
        plt.scatter(data_points[1], np.log(data_points[0]))
    else:
        plt.scatter(data_points[1], data_points[0])
    plt.title(f"Plot between {x_axis} and {y_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def xy_plot_measurement_error_plot(x_axis, y_axis, SyPaAn_data):

    data_points = transform_SyPaAn_single_measure_single_dependency(SyPaAn_data, y_axis, [x_axis])

    # transform data into one vector containing all x values and one matrix containing in the column the y values for the respective x value
    x_values = np.unique(data_points[1])
    y_values = data_points[0].reshape(len(x_values),int(len(data_points[0])/len(x_values))).transpose()

    print(y_values)
    yerr = 1.96 * np.std(y_values, axis=0)
    y_vals_mean = np.mean(y_values, axis=0)
    print(yerr)

    fig = plt.figure()

    plt.errorbar(x_values, y_vals_mean, yerr=yerr)
    plt.title(f"Plot between {x_axis} and {y_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()


def plot_beta_against_max_var_two_sims(x_axis, y_axis, SyPaAn_data_sim1, SyPaAn_data_sim2, color1, color2):

    fig = plt.figure()

    # transform data into a matrix
    data_points = transform_SyPaAn_single_measure_single_dependency(SyPaAn_data_sim1, y_axis, [x_axis])

    # transform data into one vector containing all x values and one matrix containing in the column the y values for the respective x value
    x_values = np.unique(data_points[1])
    y_values = data_points[0].reshape(len(x_values),int(len(data_points[0])/len(x_values))).transpose()

    y_vals_mean = np.mean(y_values, axis=0)
    yerr_upper  = y_vals_mean + 1.96 * np.std(y_values, axis=0)
    yerr_lower  = y_vals_mean - 1.96 * np.std(y_values, axis=0)

    plt.plot(x_values, y_vals_mean, color = color1,
             label=fr"{SyPaAn_data_sim1[0]['model_type']} Model with $M = {SyPaAn_data_sim1[0]['M']}$ and "
                   fr"$N = {SyPaAn_data_sim1[0]['no_of_agents']}$")
    plt.fill_between(x_values, yerr_lower, yerr_upper, color=color1, alpha = 0.5)


    # transform data into a matrix
    data_points = transform_SyPaAn_single_measure_single_dependency(SyPaAn_data_sim2, y_axis, [x_axis])

    # transform data into one vector containing all x values and one matrix containing in the column the y values for the respective x value
    x_values = np.unique(data_points[1])
    y_values = data_points[0].reshape(len(x_values),int(len(data_points[0])/len(x_values))).transpose()

    y_vals_mean = np.mean(y_values, axis=0)
    yerr_upper  = y_vals_mean + 1.96 * np.std(y_values, axis=0)
    yerr_lower  = y_vals_mean - 1.96 * np.std(y_values, axis=0)

    plt.plot(x_values, y_vals_mean, color = color2,
             label=fr"{SyPaAn_data_sim2[0]['model_type']} Model with $M = {SyPaAn_data_sim2[0]['M']}$ and "
                   fr"$N = {SyPaAn_data_sim2[0]['no_of_agents']}$")

    plt.fill_between(x_values, yerr_lower, yerr_upper, color=color2, alpha = 0.3)


    plt.title("Plot between $\\beta$ and maximum observed Variance")
    plt.xlabel(r"$\beta$")
    plt.ylabel("maximum Variance")
    plt.legend(loc="upper left")

    plt.show()

def two_d_histogramm_single_simulation(matr, NO_OF_BINS, C):
    """
    Create a single plot that shows the change in attitude distribution over time with the use of histograms.

    :param matr: 2d matrix containing an attitude distribution for each iteration (iterations are the columns)
    :param NO_OF_BINS: Number of bins that will be used for calculating the histogram weights
    :param C: either a connection matrix (normal ArgModel) or the number of implicitly modelled arguments (reduced ArgModel)
    :return: Nothing, but a matplotlib plot is shown
    """

    # set size of final plot
    fig = plt.figure(figsize=(15,3))
    fontsize = 8

    # calulates the maximal and minimal possible attitude based on C
    if type(C) is int:
        no_modelled_args_per_side = C
        lims = np.array([[-1,1]])
    else:
        lims = calc_min_max_atts(C)

    cmap_out_group = mpl.cm.Reds
    cmap_out_group.set_under(color='white')

    data = distr_to_2d_histogram(matr, NO_OF_BINS, lims[0,:])
    extent = [0, data.shape[1], lims[0,0], lims[0,1]]

    plt.imshow(data, extent = extent, cmap=cmap_out_group, interpolation='None', aspect='auto', vmin=0.001)
    plt.title(f"Distribution over time", fontsize=fontsize)
    plt.ylabel("Attitude")
    plt.xlabel("Time in Iterations")
    plt.grid(visible=True, axis='both', color='black', alpha=0.3)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.show()


def distr_to_2d_histogram(matr, no_of_bins, lims):
    """
    Transform a 2d matrix containing attitudes over time into a 2d matrix containing the bin weights for each iteration

    :param matr: 2d matrix containing agents attitude over time
    :param no_of_bins: No of bins with which to calculate the hist values
    :param lims: boundaries between the hist is calculated.
    :return: 2d matrix containing for each column (representing one iteration) the hist bin weights
    """

    # empty matrix that will be filled with the hist value
    hist_matr = np.zeros((no_of_bins, matr.shape[1]))

    # calculate the hist weights for each iteration
    for i in range(matr.shape[1]):
        hist_matr[:, i], edges = np.histogram(matr[:,i], bins=no_of_bins, range=lims)

    return hist_matr


def calc_min_max_atts(C):
    """
    Calculate the maximal and minimal attitude values. They are decided by the number of 1'sin the linkage matrix C.

    :param C: Connection matrix consisting of 1, 0, or -1
    :return: Matrix with the max and min possible value for each row in C
    """
    # matrix for saving the result
    max_min_atts = np.zeros((C.shape[0], 2))
    # iterate through each behaviour (the rows of C)
    for i in range(C.shape[0]):
        # save the elements in a row where C is bigger than zero.
        rowmax = np.where(C[i, :] > 0)
        # the length of the array containing those elements decides the maximally possible attitude
        max_min_atts[i, 1] = np.sum(C[i,rowmax[1]])
        rowmin = np.where(C[i, :] < 0)
        max_min_atts[i, 0] = np.sum(C[i,rowmin[1]])

    return max_min_atts


def max_mean_pairwise_distance(C):
    """
    # Calculate the maximal value possible for the pairwise distances (in 2d space, if two point are in opposite
    # corners of the grid.

    :param C: Connection matrix
    :return: Maximum pairwise distance
    """
    lmts = calc_min_max_atts(C)
    max_mean_dist = np.linalg.norm(lmts[:, 1])
    return max_mean_dist


def calc_means(matr):
    """
    Calculate the mean attitude for each iteration

    :param matr: matrix with attitude distribution over time
    :return: means over time
    """
    means = np.zeros(matr.shape[0])

    for i in range(matr.shape[0]):
        means[i] = np.mean(matr[i, :], axis=0)

    return means

"""
***********************************************************************************************************************
THE FOLLOWING FUNCTIONS MIGHT NOT WORK AS INTENDED. CHECK BEFORE USE!!!
***********************************************************************************************************************
"""

def calc_mean_distance_time(atts_over_time):
    """
    Calculate the mean distance of attitude from the mean attitude over time

    :param atts_over_time:
    :return:
    """

    for ele in atts_over_time:
        ele['attitudes'] = rearrange_attitude_list(ele['attitudes'])
        ele.update({'MD': calc_variance(ele['attitudes'])[0] })

    return atts_over_time

# calculates the bias for a given set of distributions of attitudes
def calc_bias(matr):
    bias_dist = []
    for i in range(matr.shape[0]):

        bias_dist.append(np.linalg.norm(np.mean(matr[i, :, :], axis=0)))

    return bias_dist

# calculates the bias for attitudes over time
def calc_bias_distance_time(atts_over_time):

    for ele in atts_over_time:
        ele['attitudes'] = rearrange_attitude_list(ele['attitudes'])
        ele.update({'BI': calc_bias(ele['attitudes'])[1] })

    return atts_over_time


def calc_complete_linkage(matr, agents_in_group):
    compl_link = []

    for i in range(matr.shape[0]):

        mask = np.ones(matr.shape[1], dtype=bool)
        mask[agents_in_group] = False
        distance = ssd.cdist(matr[i, agents_in_group, :], matr[i, mask, :])
        compl_link.append(np.mean(distance))

    return compl_link

# calulates the mean pairwise distance for a set of attitudes
def mean_pairwise_distance(matr):
    means = []
    for i in range(matr.shape[0]):

        pw_distances = pairwise_distances(matr[i,:,:])
        mean = np.mean(pw_distances)
        means.append(mean)

    return means

# calculates the mean pairwise distance for attitudes over time
def calc_mean_pairwise_over_time(atts_over_time):

    for ele in atts_over_time:
        ele['attitudes'] = rearrange_attitude_list(ele['attitudes'])
        ele.update({'Esteban': mean_pairwise_distance(ele['attitudes']) })

    return atts_over_time


# this is needed for proper data structure and data accessability. Is converts a list of 2d arrays into a 3d array.
def rearrange_attitude_list(list_of_attitude_list):
    matr = np.array(list_of_attitude_list)
    return matr

def plot_iterations_DEPRECATED(opinions_over_time, no_of_bins, lmts):
    """
    DEPRECATED
    Convert a list of an array into a list of histograms

    :param opinions_over_time: list of opinion distributions
    :param no_of_bins: Number of bins for hist weight calculation
    :param lmts: boundaries for hist calculation
    :return: list of histogram weights and the extent of the calculated data
    """

    # sets the upper and lower limits for the y-axis
    spread = np.floor(lmts[1])-np.ceil(lmts[0])

    stepsize = spread/no_of_bins

    # to show the data in a 2d heatmap, they are written into a 2d array
    data_in_proper_form = np.zeros((no_of_bins, len(opinions_over_time)))
    hel = len(data_in_proper_form)

    #restructuring data to allow for the heatmap to be generated
    for i in range(len(opinions_over_time)):
        for j in range(hel):
            data_in_proper_form[j][i] = count_no_of_occurence_in_intervall(np.ceil(lmts[0]), (np.floor(lmts[1]))-stepsize*(j+1), (np.floor(lmts[1]))-stepsize*j,
                                                                           opinions_over_time[i])

    return data_in_proper_form, [0 , len(opinions_over_time), lmts[0] , lmts[1]]


def count_no_of_occurence_in_intervall_DEPRECATED(llb, lb, ub, arr):
    """
    Count the number of times an opinion within a given intervall occurs in a list of opinions

    :param llb: the absolutely lowest value possible for all function calls
    :param lb: lower boundary of the interval
    :param ub: upper boundary of the interval
    :param arr: numbers in an array
    :return: Number of elements within the given interval
    """
    no=0

    # exception for the lowest possible lb. Otherwise some values would either be left out or counted twice
    if lb == llb:
        for ele in arr:
            if lb <= ele <= ub:
                no += 1
    else:
        for ele in arr:
            if lb < ele <= ub:
                no += 1
    return no
