import ArgModel as normal_am
import reducedArgModel as reduced_am

import numpy as np

# defines the parameters needed to run the model
params = {
    "no_of_agents": 100,
    "no_of_iterations": 1000,
    # number of evaluations
    "M": 8,
    # strength of biased processing
    "ß": 0,
    # binary variable indicating what data to return at the end of the simulation
    "SPA": False}

# simulates a model run and saves the returned data for later use
loal = reduced_am.simulate_agent_interaction(params["no_of_agents"], params["no_of_iterations"], params["M"],
                                             params["ß"], params["SPA"])

print(loal[100])

# defines the parameters needed to run the model
params = {
    "no_of_agents": 100,
    "no_of_iterations" : 100000,
    # strength of biased processing
    "ß": 3.2,
    # linkage matrix
    "C": np.asmatrix([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64),
    # binary variable indicating what data to return at the end of the simulation
    "SPA": False}

loal, lovl = normal_am.simulate_agent_interaction(params["no_of_agents"], params["no_of_iterations"], params["ß"],
                                                  params["C"], params["SPA"])

