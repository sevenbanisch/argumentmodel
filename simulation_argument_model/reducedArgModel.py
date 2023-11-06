# import needed packages
import numpy as np
from numba import jit
import utilities_simulation as us


@jit(nopython=True)
def initiate_agents(no_agents):
    """
    Initiate no_agents agents with uniform random attitudes between -1 and 1 and returns them in a matrix.

    :param no_agents: Number of agents
    :return: list of agents attitude
    """

    agent_att = np.random.uniform(-1, 1, no_agents)

    return agent_att


@jit(nopython=True)
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


@jit(nopython=True)
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

    no_of_agents = model_parameters["no_of_agents"]
    no_of_iterations = model_parameters["no_of_iterations"]
    beta = model_parameters["ß"]
    M = model_parameters["M"]
    if type(M) is not int:
        raise ValueError("The number of implicitely modelled arguments M must be of type int")

    SyPaAn = model_parameters["SPA"]

    # initiates the agents
    agents_att = initiate_agents(no_of_agents)
    agent_indices = np.arange(0, no_of_agents)

    # simulates a single iteration
    for interaction in range(no_of_iterations):

        np.random.shuffle(agent_indices)
        agents_att = single_interaction(agents_att, agent_indices, beta, M)

        # data about the simulation run is collected and stored for later analysis. It is only stored after a
        # "Macro-iteration", meaning after no_of_agents iteration.
        measures = us.update_measure_dict(None, agents_att, interaction, measures)

    # if a Systematic Parameter Analysis is performed, only the state of the agents
    # after the last iteration is of concern
    if SyPaAn:
        measures = us.update_measure_dict_for_SyPaAn(None, agents_att, no_of_iterations, measures)
        # returns the attitude at the end of the model simulation and the indexes of agents in the group
        return measures

    # returns the list of attitudes for each iteration, the list of evaluations for each iteration and
    # the indexes of the agents in the group
    return measures
