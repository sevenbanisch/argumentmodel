# import needed packages
import numpy as np
from numba import jit
from . import utilities_simulation as us
import itertools
import copy


@jit(nopython=True)
def initiate_agents(no_agents, C, initiation_type):
    """
    Initiate no_agents agents with uniform random attitudes between -1 and 1 and returns them in a matrix.

    :param no_agents: Number of agents
    :param C: linkage matrix
    :param initiation_type: indicator describing in which way to initiate the agents attitude
    :return: list of agents attitude
    """

    if initiation_type =="uniform":
        agent_att = np.reshape(np.random.uniform(-1, 1, no_agents), (no_agents, C.shape[0])).astype(np.float64)
    elif initiation_type == "binomial":
        agent_eval = np.reshape(np.random.randint(0, 2, no_agents * C.shape[1]), (no_agents, C.shape[1])).astype(np.float64)
        agent_att = agent_eval @ C.transpose()
    else:
        raise ValueError("Please pick a different initiation type")

    return agent_att


@jit(nopython=True, fastmath=True)
def expected_change_in_opinion(M, beta, att_sender, att_receiver):
    """
    Calculation of expected change of opinion (see equation 10 in paper draft of reduced arg model)

    :param M: Number of underlying arguments
    :param beta: Strength of biased processing
    :param att_sender: list/scalar of attitudes of the senders
    :param att_receiver: list/scalar of attitudes of the receivers
    :return: list of expected attitude change for all receivers based on their respective sender
    """

    # Prefactor correcting possible strength of attitude adoption
    prefactor = (1/(2*M))
    diff_in_att = att_sender - att_receiver
    tan_hyp = np.tanh(att_receiver * beta * 0.5)
    # prefactor to the tangens hyperbolicus
    multiplier = (1-np.multiply(att_sender, att_receiver))

    d_opinion = prefactor * (diff_in_att + np.multiply(multiplier, tan_hyp))

    return d_opinion


@jit(nopython=True, fastmath=True)
def single_interaction(agents_att, agent_indices, beta, M):
    """
    Implement a single iteration in the given model

    :param agents_att: Matrix of attitudes for all agents
    :param agent_indices: shuffles list of agent indices to match agents randomly
    :param beta: strength of biased processing
    :param M: Number of arguments
    :return: The new Attitude Matrix for all agents
    """
    no_of_agents = len(agents_att)
    agents_midpoint = int(no_of_agents / 2)

    receivers_indices = agent_indices[0:agents_midpoint]
    senders_indices = agent_indices[agents_midpoint:no_of_agents]

    # attitudes of the receivers and senders
    agents_att_receivers = agents_att[receivers_indices]
    agents_att_senders = agents_att[senders_indices]

    d_opinion = expected_change_in_opinion(M, beta, agents_att_senders, agents_att_receivers)

    # adoption of the expected attitude chang
    agents_att[receivers_indices] = agents_att_receivers + d_opinion
    agents_att = np.clip(agents_att, -1, 1)

    return agents_att


def simulate_agent_interaction(model_parameters, measures):
    """
    Simulate the whole model. If SyPaAn is True, only the state of the model after the last iteration will be returned.

    :param model_parameters: Model parameters such as number of agents etc.
    :param measures: dict with measures to be taken and a variable in which to add the taken measurements
    :return:
        loal: list of attitude distribution throughout the iterations
        measures: dictionary with taken measures in a list
    """

    no_of_agents = int(model_parameters["no_of_agents"])
    no_of_iterations = model_parameters["no_of_iterations"]
    beta = model_parameters["ß"]
    M = int(model_parameters["M"])
    no_of_iterations = int(M/4+0.5) * 1000
    implied_C = us.create_connection_matrix_symmetrical(M, True)
    if type(M) is not int:
        raise ValueError("The number of implicitly modelled arguments M must be of type int")

    # initiates the agents
    agents_att = initiate_agents(no_of_agents, implied_C, model_parameters["initiation"])
    agent_indices = np.arange(0, no_of_agents)

    # simulates a single iteration
    for interaction in range(no_of_iterations):

        np.random.shuffle(agent_indices)
        agents_att = single_interaction(agents_att, agent_indices, beta, M)

        # data about the simulation run is collected and stored for later analysis.
        measures, stop_sim = us.update_measure_dict_during_simulation(None, agents_att, interaction, measures, model_parameters)
        if stop_sim:
            for not_modelled in range(interaction+1, no_of_iterations):
                measures, stop_sim = us.update_measure_dict_during_simulation(None, agents_att, not_modelled, measures, model_parameters)
            break


    measures = us.update_measure_dict_after_simulation(None, agents_att, no_of_iterations, measures)

    # returns the list of attitudes for each iteration, the list of evaluations for each iteration and
    # the indexes of the agents in the group
    return measures



# implements the iteration through a predefined parameter space
def systematic_parameter_analysis(SPA_params, params, measures):

    # list which will contain the results
    measures_from_SPA = []

    params_possbls = []
    # the parameter values that are iterated over are created using the upper and lower boundary provided by a variable, and the provided number of steps for each parameter
    for i in range(len(SPA_params['params_to_iter'])):
        params_possbls.append(np.linspace(SPA_params['boundaries'][i, 0], SPA_params['boundaries'][i, 1], SPA_params['no_of_steps'][i]))

    # creates the cartesion product out of the parameter values
    cartesian = itertools.product(*params_possbls)

    # runs a certain number of simulations for every parameter combination
    for ele in cartesian:
        print(np.round(ele,2))
        measures_from_single_comb = []
        for index, value in enumerate(ele):
            params[SPA_params['params_to_iter'][index]] = value

        for i in range(SPA_params['sims_per_comb']):
            measures_single_sim = copy.deepcopy(measures)
            # runs the model and returns the attitudes after the last iteration, as well as the inidices of the group members
            measures_single_sim = simulate_agent_interaction(params, measures_single_sim)
            measures_from_single_comb.append(measures_single_sim.copy())

        # saves the results in a dictionary
        dict_comb = {k: [d[k] for d in measures_from_single_comb] for k in measures_from_single_comb[0]}
        dict_comb.update(params)
        dict_comb.update({"model_type": "Reduced"})

        # adds the dictionary to the results list
        measures_from_SPA.append(dict_comb)

    return measures_from_SPA
