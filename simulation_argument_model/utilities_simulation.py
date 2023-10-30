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

    measure_name = "variance_attitude"
    if measure_name in measures_to_be_taken:
        measures[measure_name][interaction] = np.std(agents_att)**2

    measure_name = "correlation_evaluations"
    if measure_name in measures_to_be_taken:
        raise Exception("Not yet implemented :)")

    return measures
