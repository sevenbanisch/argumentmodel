# import needed packages
import numpy as np
from numba import jit


@jit(nopython=True)
def initiate_agents(no_agents, no_arguments):
    """
    Initiate no_agents agents with no_arguments uniform random evaluations of outcomes and returns them in a matrix.

    :param no_agents: Number of agents
    :param no_arguments: Number of arguments
    :return: no_agents x no_arguments matrix filled with boolean evaluations of the arguments
    """

    agent_eval = np.reshape(np.random.randint(0, 2,no_agents* no_arguments), (no_agents, no_arguments)).astype(np.float64)

    return agent_eval


@jit(nopython=True)
def calculate_attitude_of_agents(agents, C):
    """
    Calculate the attitude and norm of all agents.

    :param agents: matrix of agents evaluations of arguments
    :param C: Connection matrix
    :return: matrix with attitudes for all agents
    """
    return np.dot(agents, C.transpose())


@jit(nopython=True)
def p_beta_diff(beta, coherence_diff):
    """
    Calculate the probability of argument adoption for all coherence difference supplied.

    :param beta: strength of biased processing
    :param coherence_diff: difference in coherence before and after argument adoption for each receiver
    :return: probability of argument acceptance for each receiver
    """
    res = 1/(1+np.exp(beta * (coherence_diff)))
    return res


@jit(nopython=True)
def part_of_eval_matrix(agents_eval, indexes):
    """
    Return part of the evaluation matrix, only keeping the provided indices

    :param agents_eval: Matrix containing agents evaluations
    :param indexes: List of indices that will be return
    :return: Matrix of agents argument evaluations
    """
    res = 2.0 * agents_eval[indexes] - 1.0
    return res


@jit(nopython=True)
def coherence_diff(C_selective, agents_att_receivers, arg_diff):
    """
    calculation of the coherence difference: (arg_sender - arg_receiver) * e_i * (attitude) for *each* receiver at once

    :param C_selective: Polarisations of the communicated arguments
    :param agents_att_receivers: attitudes of the receivers
    :param arg_diff: difference in the evaluation of the communicated argument between receiver and sender
    :return: difference in coherence from argument adoption for all receivers
    """
    res = np.multiply(np.multiply(C_selective.transpose(), agents_att_receivers), arg_diff.transpose())
    return res


@jit(nopython=True)
def single_interaction(agents_eval, agents_att, agent_indices, beta, C, communicated_arguments, random_numbers):
    """
    Implement a single iteration in the given model

    :param agents_eval: Matrix of evaluations for each agent
    :param agents_att: Matrix of attitudes for all agents
    :param agent_indices: shuffles list of agent indices to match agents randomly
    :param beta: strength of biased processing
    :param C: Connection matrix
    :param communicated_arguments: Argument a sender communicates
    :param random_numbers: list of random numbers that are used to decide if an agent adopts an argument
    :return: The new Evaluation Matrix for all agents
    """
    no_of_agents = len(agents_eval)
    agents_midpoint = int(no_of_agents / 2)

    # this array will store the polarisation of the communicated argument for each receiver
    C_selective = np.zeros(agents_midpoint)

    receivers_indices = agent_indices[0:agents_midpoint]
    senders_indices = agent_indices[agents_midpoint:no_of_agents]

    # placeholder that will be filled with the arguments of the sender (-> new) and receiver (-> old)
    arg_old = np.zeros(agents_midpoint)
    arg_new = np.zeros(agents_midpoint)

    # Evaluations of the receivers and senders
    agents_eval_receivers = agents_eval[receivers_indices]
    agents_eval_senders = agents_eval[senders_indices]

    # extracts the evaluation of the communicated argument and its respective polarisation
    for index, argument in enumerate(communicated_arguments):
        # The polarity of the argument a sender has chosen to communicate
        C_selective[index] = np.round(C[0, argument], 4)
        # The evaluation of the argument for the respective pairing
        arg_old[index] = agents_eval_receivers[index, argument]
        arg_new[index] = agents_eval_senders[index, argument]

    # attitudes of the receivers
    agents_att_receivers = agents_att[receivers_indices, 0]

    arg_diff = arg_old - arg_new

    coherence_difference = coherence_diff(C_selective, agents_att_receivers, arg_diff)
    # row-wise calculation of the probability of argument acceptance
    p_beta_difference = p_beta_diff(beta, coherence_difference)

    # receiver adopts the communicated argument with a probability of p_beta_diff
    for no, agent_index in enumerate(receivers_indices):
        if random_numbers[no] < p_beta_difference[no]:
            agents_eval[agent_index, communicated_arguments[no]] = agents_eval[senders_indices[no], communicated_arguments[no]]

    return agents_eval


def simulate_agent_interaction(no_of_agents, no_of_iterations, beta, C, SyPaAn):
    """
    Simulate the whole model. If SyPaAn is True, only the state of the model after the last iteration will be returned.

    :param no_of_agents: Number of agents simulated in the model
    :param no_of_iterations: Number of Iterations
    :param beta: Strength of biased processing
    :param C: Connection Matrix used to connect arguments to an agents attitude
    :param SyPaAn: Bool indicating wether to return data for more than one iteration or only for the end of a model run
    :return:
        loal: list of attitude distribution throughout the iterations
        lovl: list of evaluation distributions throughout the iterations

    """

    # Only if we are not conduction a Systematic Parameter Analysis will we need these lists
    if not SyPaAn:
        list_of_attitude_lists = []
        list_of_eval_lists = []

    # initiates the agents
    agents_eval = initiate_agents(no_of_agents, C.shape[1])

    # calculates initial attitude of all agents
    agents_att = calculate_attitude_of_agents(agents_eval, C)

    agents_midpoint = int(no_of_agents/2)

    agent_indices = np.arange(0, no_of_agents)

    #simulates a single iteration
    for interaction in range(no_of_iterations):

        np.random.shuffle(agent_indices)

        l_communicated_argument = np.random.randint(0, C.shape[1]+1, agents_midpoint)
        l_random_number_for_if_clause = np.random.uniform(0, 1, agents_midpoint)

        agents_eval = single_interaction(agents_eval, agents_att, agent_indices, beta, C, l_communicated_argument, l_random_number_for_if_clause)

        agents_att = calculate_attitude_of_agents(agents_eval, C)

        # data about the simulation run is collected and stored for later analysis. It is only stored after a
        # "Macro-iteration", meaning after no_of_agents iteration.
        if not SyPaAn:
            list_of_eval_lists.append(agents_eval.copy())
            list_of_attitude_lists.append(agents_att.copy())

    # if a Systematic Parameter Analysis is performed, only the state of the agents
    # after the last iteration is of concern
    if SyPaAn:
        # returns the attitude at the end of the model simulation and the indexes of agents in the group
        return agents_att.copy()

    # returns the list of attitudes for each iteration, the list of evaluations for each iteration and the indexes of the agents in the group
    return list_of_attitude_lists, list_of_eval_lists