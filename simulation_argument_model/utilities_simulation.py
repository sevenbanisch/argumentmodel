import numpy as np


def update_measure_dict(agents_eval, agents_att, interaction, measures):
    """
    Calculate all measures defined in measures.keys() for the given iteration.
    The measures are added into the array given in the measures dict

    :param agents_eval: evaluation of arguments for all agents numpy array. Is None in the reduced argument model
    :param agents_att: attitudes of all agents in a numpy array
    :param interaction: The current interation as an integer
    :param measures: dictionary containing the measures that are to be taken
    :return:
        measures: the update version of the measures, that updated the arrays corresponding to the keys
    """

    measures_to_be_taken = list(measures.keys())

    measure_name = "attitude_of_all_agents"
    if measure_name in measures_to_be_taken:
        measures[measure_name][:, interaction] = agents_att.reshape((len(agents_att),))

    measure_name = "mean_attitude"
    if measure_name in measures_to_be_taken:
        measures[measure_name][interaction] = np.mean(agents_att)

    measure_name = "mean_attitude_END_VALUE"
    if measure_name in measures_to_be_taken:
        measures[measure_name] = np.mean(agents_att)

    measure_name = "variance_attitude"
    if measure_name in measures_to_be_taken:
        measures[measure_name][interaction] = np.std(agents_att)**2

    measure_name = "variance_attitude_END_VALUE"
    if measure_name in measures_to_be_taken:
        measures[measure_name] = np.std(agents_att)**2

    measure_name = "correlation_of_evaluations"
    if measure_name in measures_to_be_taken:
        measures[measure_name][:, :, interaction] = np.cov(agents_eval)

    return measures


def update_measure_dict_for_SyPaAn(agents_eval, agents_att, no_of_iterations, measures):
    return None


def create_connection_matrix_symmetrical(no_of_arguments, normalised):
    """
    Create a symmetrical Connection matrix in which half of the arguments are considered pro and the other half con.
    Can optionally be normalised so that the values of the matrix are divided by M*0.5
    :param no_of_arguments: Number of arguments (columns of the connection matrix)
    :param normalised: Boolean if the matrix will be normalised
    :return:
        C: symmetrical connection matrix
    """
    C = np.ones(no_of_arguments)
    midpoint_of_C = int(len(C) * 0.5)
    C[midpoint_of_C:] = - 1
    C = np.asmatrix(C, dtype=np.float64)

    if normalised:
        return C / (no_of_arguments * 0.5)
    else:
        return C
