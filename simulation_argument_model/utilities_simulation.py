import numpy as np
from numba import jit


def update_measure_dict(agents_eval, agents_att, interaction, measures):

    measures_to_be_taken = list(measures.keys())

    measure_name = "attitude_all_agents"
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
        measure_name["hallo"] = 1
        raise Exception("Not yet implemented :)")

    return measures
