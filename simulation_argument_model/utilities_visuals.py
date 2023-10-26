# import of necessary libraries
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as ssd


# counts number of times an opinion within a given intervall occurs in a list of opinions
def count_no_of_occurence_in_intervall(llb, lb, ub, arr):
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


# calculated the maximal and minimal attitude values. They are decided by the number of 1's in the linkage matrix C.
def calc_min_max_atts(C):
    # matrix for saving the result
    max_min_atts = np.zeros((C.shape[0], 2))
    # iterate through each behaviour (the rows of C)
    for i in range(C.shape[0]):
        # save the elements in a row where C is bigger than zero.
        rowmax = np.where(C[i, :] > 0)
        # the length of the array containing those elements decides the maximally possible attitude
        max_min_atts[i, 1] = len(rowmax[0])
        rowmin = np.where(C[i, :] < 0)
        max_min_atts[i, 0] = -len(rowmin[0])

    return max_min_atts


# calculated the maximal value possible for the pairwise distances (in 2d space, if two point are in opposite
# corners of the grid.
def max_mean_pairwise_distance(C):
    lmts = calc_min_max_atts(C)
    max_mean_dist = np.linalg.norm(lmts[:, 1])
    return max_mean_dist


# calculates the mean attitude for all iterations
def calc_means(matr):
    means = []
    for i in range(matr.shape[0]):
        means.append(np.mean(matr[i, :, :], axis=0))

    return means


# calculates the mean distance of attitude from the mean attitude over time
def calc_mean_distance_time(atts_over_time):

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