import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


# erzeugt einen zufälligen agenten gemäß der übergebenen parameter und stattet ihn mit einem index aus
def initiate_random_agent(M, index):
    array_pro = []
    array_con = []
    # der Agent wird mit M zufälligen pro- und contra Argumenten initialisiert
    for i in range(int(M)):
        num_pro = random.randint(0, 1)
        array_pro.append(num_pro)
        num_con = random.randint(0, 1)
        array_con.append(num_con)
    agent = {'index': index, 'a_pro': np.array(array_pro), 'a_con': np.array(array_con)}
    # berechnen der ursprünglichen opinion
    agent.update({'opinion': sum(agent['a_pro'])-sum(agent['a_con'])})
    return agent


# bestimmt die evaluation des arguments, welche auf der Köhärenz aufbaut
def calculate_individual_evaluation_of_argument(agent, isPro, isOne):
    if isOne == 0:
        isOne = -1
    # Je nachdem ob es sich um ein pro- oder contraargument handelt, muss das Vrozeichen umgedreht werden
    if isPro:
        return isOne * (sum(agent['a_pro'])-sum(agent['a_con']))
    else:
        return isOne * (sum(agent['a_con'])-sum(agent['a_pro']))


# berechnet die Wahrscheinlichkeit, mit der ein argument übernommen wird
def calculate_probability_of_acceptance(agent, isPro, beta, isOne):
    evaluation = calculate_individual_evaluation_of_argument(agent, isPro, isOne)
    # Berechnung der Sigmoid funktion zur Berechnung der Wahrscheinlichkeit
    return 1/(1+math.exp(-(beta * evaluation)))


# berechnet die Opinion eines einzelnen Agenten
def calculate_opinion_of_agent(agent):
    return sum(agent['a_pro'])-sum(agent['a_con'])


# berechnet die opinion einer liste von agenten
def calculate_opinion_of_list(list_of_agents):
    for agent in list_of_agents:
        agent['opinion'] = calculate_opinion_of_agent(agent)
    return list_of_agents


# erstellt eine heatmap der opinions und wie diese sich über zeit entwickeln
def plot_iterations(opinions_over_time):
    # damit die Daten in einer 2d-heatnmap angezeigt werden können, werden sie in ein 2d array geschrieben
    data_in_proper_form = np.zeros((9, len(opinions_over_time)))
    for i in range(len(opinions_over_time)):
        for j in range(4, -5, -1):
            # die Indexverschiebungen sorgen für die korrekten zahlen an der richtigen Stelle
            data_in_proper_form[j + 4][i] = opinions_over_time[i].count((-1 * j))

    plt.imshow(data_in_proper_form, aspect='auto', interpolation='none')
    plt.title("Distribution of opinions over the course of Iterations")
    plt.colorbar()
    plt.show()


# gibt die opinions einer liste von agenten in einer liste zurück.
# Anhand dieser Liste von opinions können verteilungen geschätzt werden
def get_distribution_of_list_of_agents(list_of_agents):
    opinion_list = []
    for i in range(len(list_of_agents)):
        opinion_list.append(list_of_agents[i]['opinion'])

    return opinion_list


# erstellt eine liste von no_of_agents agenten mit zufälligen geglaubten argumenten
def iniate_list_of_agents(no_of_agents, M):
    list_of_agents = []
    for i in range(no_of_agents):
        agent = initiate_random_agent(M, i)
        list_of_agents.append(agent)
    return list_of_agents


# die beiden folgenden funktionen beschreiben die interaktion auf ebene eines agentenpaares (receiver/sender)
# da es Unterschiede in der Berechnung gibt, je nach dem ob es sich um ein pro oder con argument handelt werden
# zwei funktionen benutzt. Es wäre auch möglich das mit einer if-schleife zu lösen

def simulate_interaction_on_individual_level_pro(receiver, sender, beta, new_arg):
    # indexanpassung
    new_arg = new_arg - 1
    # Interaktion kann nur stattfinden, wenn die Agenten eine unterschiedliche Meinung haben
    if receiver['a_pro'][new_arg] != sender['a_pro'][new_arg]:
        isOne = sender['a_pro'][new_arg]
        probability_to_accept = calculate_probability_of_acceptance(receiver, True, beta, isOne)
        # receiver übernimmt das kommunizierte argument mit einer gewissen wahrscheinlichkeit
        if random.random() <= probability_to_accept:
            receiver['a_pro'][new_arg] = sender['a_pro'][new_arg]

    return receiver['a_pro']


def simulate_interaction_on_individual_level_con(receiver, sender, beta, new_arg, M):
    # indexanpassung
    new_arg = new_arg - 1 - M
    # Interaktion kann nur stattfinden, wenn die Agenten eine unterschiedliche Meinung haben
    if receiver['a_con'][new_arg] != sender['a_con'][new_arg]:
        isOne = sender['a_con'][new_arg]
        probability_to_accept = calculate_probability_of_acceptance(receiver, False, beta, isOne)
        # receiver übernimmt das kommunizierte argument mit einer gewissen wahrscheinlichkeit
        if random.random() <= probability_to_accept:
            receiver['a_con'][new_arg] = sender['a_con'][new_arg]

    return receiver['a_con']


# Simuliert die Interaktion zwischen den jeweiligen Agenten. Es werden zufällige Paare gebildet, welche dann individuell
# miteinander agieren.
def simulate_interaction(agent_list, beta, M):
    random.shuffle(agent_list)
    middle = int(0.5 * len(agent_list))

    # die jeweiligen Agentenpaare interagieren miteinander
    for i in range(middle):
        # das jeweils kommunizierte Argument wird zufällig ausgewählt
        communicated_argument = random.randint(1, int(2 * M))
        # die receiver und sender werden durch die Indexe festgelegt. Alle Agenten bis zur Mitte sind receiver,
        # alle Agenten ab der Hälfte sind sender. (i ist receiver, i+middle ist sender)
        if communicated_argument <= M:
            agent_list[i]['a_pro'] = simulate_interaction_on_individual_level_pro(agent_list[i], agent_list[i + middle],
                                                                                  beta, communicated_argument)
        else:
            agent_list[i]['a_con'] = simulate_interaction_on_individual_level_con(agent_list[i], agent_list[i + middle],
                                                                                  beta, communicated_argument, M)
    return agent_list


# checks if every agent has the same argument strings
# Ich bin mir nicht ganz sicher ob alle Argumente auf gleichheit überprüft werden müssen, oder es reicht zu überprüfen,
# ob alle Agenten die gleiche Meinung haben. An sich sollte es Fälle geben, in denen es einen Unterschied macht, ich
# hatte allerdings nach einigen Testdurchläufen kein einziges Mal den Fall, dass die Agenten unterschiedliche Argumente
# glauben aber trotzdem die gleiche Meinung haben. (Für die Laufzeit wäre Klarheit sehr interessant)
#
def check_for_same_arguments(list_of_agents):
    for i, a_1 in enumerate(list_of_agents):
        for a_2 in list_of_agents[i:]:
            if not (np.array_equal(a_1['a_pro'], a_2['a_pro']) and np.array_equal(a_1['a_con'], a_2['a_con'])):
                return False
    return True


# simuliert das gesamte modell. es werden die anzahl an agenten, die gewünschten iterationen,
# die Stärke des biased processing und die anzahl der argumente übergeben.
def simulate_agent_interaction(no_of_agents, no_of_iterations, beta, M, check_for_consens):
    list_of_opinion_lists = []
    list_of_agents = iniate_list_of_agents(no_of_agents, M)

    # variable checks if every agent has the same opinion
    # consens = False

    for i in tqdm(range(no_of_iterations)):
        list_of_agents = simulate_interaction(list_of_agents, beta, M)
        list_of_agents = calculate_opinion_of_list(list_of_agents)
        list_of_opinions = get_distribution_of_list_of_agents(list_of_agents)
        #consens_time = get_consens_time_in_individual_simulation(list_of_opinions)
        list_of_opinion_lists.append(list_of_opinions)
        #list_of_opinion_lists.append(consens_time)

        # checks if every agent has the same opinion
        if check_for_consens and (len(set(list_of_opinions)) == 1):
            # checks if every agent has the same argument strings
            if check_for_same_arguments(list_of_agents):
                # consens = True
                break

    consens_time = get_consens_time_in_individual_simulation(list_of_opinion_lists)
    # if (consens==1):
    #    while (len(list_of_opinion_lists) < no_of_iterations):
    #        list_of_opinion_lists.append(list_of_opinion_lists[i])

    # gibt die Zeit zurück, die benötigt wurde bis alle Agenten die gleiche Meinung haben
    return consens_time #list_of_opinion_lists


# implement the incremental change of ß. Multiple simulations are made for each ß, to get a more reliable mean from the
# simulation results
def systematic_parameter_analysis(sim_per_parameter_step, lb, ub, no_of_steps, no_of_agents, no_of_iterations, M):
    beta = lb
    results_in_matrix = []
    stepsize = (ub - lb) / no_of_steps

    while beta < ub:
        print(f"ß= {beta}")
        results_beta = []
        for i in range(sim_per_parameter_step):
            results_beta.append(simulate_agent_interaction(no_of_agents, no_of_iterations, beta, M, True))

        #results_in_matrix.append({'beta': beta, 'opinion': get_consens_time_in_simulation(results_beta)})
        results_in_matrix.append({'beta': beta, 'opinions': results_beta})
        beta += stepsize

    return results_in_matrix,


def get_consens_time_in_individual_simulation(single_simulation):
    time = len(single_simulation) + 1
    for iteration_inside_individual_simulation, individual_state_of_opinions in enumerate(single_simulation):
        # check if elements in a list are all equal (<==> opinions have converged)
        if len(set(individual_state_of_opinions)) == 1:
            if iteration_inside_individual_simulation < time:
                time = iteration_inside_individual_simulation
    if time == (len(single_simulation) + 1):
        time = -1

    return time

""" 

Diese Funktionen werden nur gebraucht wenn der Rückgabetyp von den Modellsimulationen nicht bereits die Konvergenzzeit ist.
def get_consens_time_in_list_of_simulations(list_of_simulation):
    list_of_times = []
    for single_simulation in list_of_simulation['opinions']:
        time = get_consens_time_in_individual_simulation(single_simulation)
        list_of_times.append(time)
    return list_of_times


# searches for the earliest time convergence has happenend in the results of the huge simulation.
# If the simulation has been done with check_for_consens on, there is an easier alternative
def get_time_for_consens(results):
    for result in tqdm(results):

        result.update({'consens_times': get_consens_time_in_list_of_simulations(result)})

    return results
    
"""

