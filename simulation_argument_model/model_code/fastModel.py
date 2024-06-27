
import numpy as np
from numba import jit

# simulates the explicit model
@jit(nopython=True, fastmath=True)
def explicit_model(M, N, T, ß, C):

    args = 2 * M

    # initiate the agents
    agent_eval = np.random.randint(0, 2, (N, args)).astype(np.float64)

    # calculate the initial attitudes
    agents_att = np.round(agent_eval @ C, 4)

    # initiate the results matrix
    results = np.empty((N, T))

    for interaction in range(T):

        # runtime optimization
        if np.var(agents_att) < 0.001:
            results[:, interaction] = agents_att
            continue

        indices = np.random.permutation(N)

        for pair in np.arange(0, N, 2):

            ind_rec = indices[pair + 1]
            ind_sen = indices[pair]

            communicated_arg = np.random.randint(args)

            sender_arg = agent_eval[ind_sen, communicated_arg]
            receiver_arg = agent_eval[ind_rec, communicated_arg]

            diff_coh = (sender_arg - receiver_arg) * agents_att[ind_rec] * C[communicated_arg] * M

            p_adopt = 1/(1+np.exp(-diff_coh * ß))
            if np.random.random() < p_adopt:

                agent_eval[ind_rec, communicated_arg] = sender_arg

        agents_att = np.round(agent_eval @ C, 4)
        results[:, interaction] = agents_att

    return results


@jit(nopython=True, fastmath=True)
def reduced_model(M, N, T, ß, C):

    args = 2 * M

    # initiate the agents
    agent_eval = np.random.randint(0, 2, (N, args)).astype(np.float64)

    # calculate the initial attitudes
    agents_att = np.round(agent_eval @ C, 4)

    # initiate the results matrix
    results = np.empty((N, T))

    for interaction in range(T):

        # runtime optimization
        if np.var(agents_att) < 0.001:
            results[:, interaction] = agents_att
            continue

        indices = np.random.permutation(N)

        for pair in np.arange(0, N, 2):

            ind_rec = indices[pair + 1]
            ind_sen = indices[pair]

            att_rec = agents_att[ind_rec]
            att_sen = agents_att[ind_sen]

            d_att = (att_sen - att_rec + np.tanh(att_rec*ß*0.5)*(1-att_rec*att_sen)) / (4*M)

            agents_att[ind_rec] += d_att

        results[:, interaction] = agents_att

    return results


@jit(nopython=True)
def SystematicParameterAnalysis(ß_info, M_steps, no_of_simulations, N, base_iterations):

    ß_steps = ß_info[2]

    # the three is because the variance is measured at three different points of the simulation.
    results_red = np.empty((ß_steps, M_steps, no_of_simulations, base_iterations))
    results_exp = np.empty((ß_steps, M_steps, no_of_simulations, base_iterations))

    for base_iteration in range(base_iterations):

        # if 3, then the base iterations 250, 1000, 4000 are iterated over
        iteration = 250 * (4**(base_iteration))

        for ß_step in range(ß_steps):

            ß = np.round((ß_step/ß_steps) * (ß_info[1] - ß_info[0]),2)

            for M_step in range(M_steps):

                M = 4**(M_step+1)

                T = M * iteration

                C = create_connection_array_symmetrical(M, True)

                for sim in range(no_of_simulations):
                    #print(f"T:{T}, N:{N}, ß:{ß}, M:{M}, C:{C}")
                    one_run_red = explicit_model(M=M, N=N, T=T, ß=ß, C=C)
                    one_run_exp = explicit_model(M=M, N=N, T=T, ß=ß, C=C)

                    results_red[ß_step, M_step, sim, base_iteration] = np.var(one_run_red[:, -1])
                    results_exp[ß_step, M_step, sim, base_iteration] = np.var(one_run_exp[:, -1])

    return results_red, results_exp

@jit(nopython=True, fastmath=True)
def create_connection_array_symmetrical(no_of_arguments, normalised):
    """
    Create a symmetrical Connection matrix in which half of the arguments are considered pro and the other half con.
    Can optionally be normalised so that the values of the matrix are divided by M*0.5. This leads to attitude values
    between -1 and 1.
    :param no_of_arguments: Number of arguments (columns of the connection matrix)
    :param normalised: Boolean if the matrix will be normalised
    :return:
        C: symmetrical connection matrix
    """
    C = np.ones(no_of_arguments*2)
    C[no_of_arguments:] = -1
    C = np.asarray(C, dtype=np.float64)

    if normalised:
        return C / (no_of_arguments)
    else:
        return C