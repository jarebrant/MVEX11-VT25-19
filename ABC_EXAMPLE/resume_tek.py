import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pyabc
from tek_model import *
from variables import *
from scipy.stats import truncnorm
from sklearn.metrics import jaccard_score


import signal


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Simulation timed out!")

signal.signal(signal.SIGALRM, timeout_handler) 

# Parameters
pp_mu = 0.05
pp_sigma = 0.1
pp_lower, pp_upper = 0, 1

# Standardized bounds
pp_a, pp_b = (pp_lower - pp_mu) / pp_sigma, (pp_upper - pp_mu) / pp_sigma

# Parameters
m_mu = 0.8
m_sigma = 0.2
m_lower, m_upper = 0, 1

# Standardized bounds
m_a, m_b = (m_lower - m_mu) / m_sigma, (m_upper - m_mu) / m_sigma

# Parameters
wm_mu = 0.8
wm_sigma = 0.2
wm_lower, wm_upper = 0, 1

# Standardized bounds
wm_a, wm_b = (wm_lower - wm_mu) / wm_sigma, (wm_upper - wm_mu) / wm_sigma

# Parameters
gm_mu = 0.2
gm_sigma = 0.1
gm_lower, gm_upper = 0, 1

# Standardized bounds
gm_a, gm_b = (gm_lower - gm_mu) / gm_sigma, (gm_upper - gm_mu) / gm_sigma

# Parameters
theta_mu = 4
theta_sigma = 2
theta_lower, theta_upper = 0, 10

# Standardized bounds
theta_a, theta_b = (theta_lower - theta_mu) / theta_sigma, (theta_upper - theta_mu) / theta_sigma


# Parameters
alpha_mu = 1.5
alpha_sigma = 3
alpha_lower, alpha_upper = 0, 10

# Standardized bounds
alpha_a, alpha_b = (alpha_lower - alpha_mu) / alpha_sigma, (alpha_upper - alpha_mu) / alpha_sigma

# Parameters
a_mu = 0.025
a_sigma = 0.1
a_lower, a_upper = 0, 1

# Standardized bounds
a_a, a_b = (a_lower - a_mu) / a_sigma, (a_upper - a_mu) / a_sigma


# Parameters
b_mu = 0.7
b_sigma = 0.2
b_lower, b_upper = 0, 1

# Standardized bounds
b_a, b_b = (b_lower - b_mu) / b_sigma, (b_upper - b_mu) / b_sigma


# Parameters
k_mu = 3
k_sigma = 4
k_lower, k_upper = 0, 10

# Standardized bounds
k_a, k_b = (k_lower - k_mu) / k_sigma, (k_upper - k_mu) / k_sigma


#Create initial cell grid based on mr_data
matrix = np.loadtxt(start_intensity,delimiter=',')
cells = create_initial_tumor(matrix,eff_radie, mid_point)
grid_size = matrix.shape
observation = np.loadtxt(end_intensity,delimiter=',')


prior = pyabc.Distribution(pp=pyabc.RV("truncnorm",pp_a, pp_b, loc=pp_mu, scale=pp_sigma),
                           m = pyabc.RV("truncnorm", m_a, m_b, loc=m_mu, scale=m_sigma),
                           alpha=pyabc.RV("truncnorm",alpha_a, alpha_b, loc=alpha_mu, scale=alpha_sigma),
                           a_pm=pyabc.RV("truncnorm",a_a, a_b, loc=a_mu, scale=a_sigma),
                           b_pm= pyabc.RV("truncnorm",b_a, b_b, loc = b_mu, scale=b_sigma),
                           theta_pm=pyabc.RV("truncnorm",theta_a, theta_b, loc=theta_mu, scale=theta_sigma),
                           k_pm=pyabc.RV("truncnorm",k_a, k_b, loc=k_mu, scale=k_sigma),
                           a_mp=pyabc.RV("truncnorm",a_a, a_b, loc=a_mu, scale=a_sigma),
                           b_mp=pyabc.RV("truncnorm",b_a, b_b, loc = b_mu, scale=b_sigma),
                           theta_mp=pyabc.RV("truncnorm",theta_a, theta_b, loc=theta_mu, scale=theta_sigma),
                           k_mp=pyabc.RV("truncnorm",k_a, k_b, loc=k_mu, scale=k_sigma),
)


def model(parameter):
    try:
        signal.alarm(600)  # timeout in 10 minutes
        result = simulate(
            cells, grid_size,
            parameter["pp"], parameter["m"],
            parameter["alpha"], parameter["a_pm"], parameter["b_pm"],
            parameter["theta_pm"], parameter["k_pm"],
            parameter["a_mp"], parameter["b_mp"],
            parameter["theta_mp"], parameter["k_mp"]
        )
        signal.alarm(0)
        return {"data": np.asarray(result, dtype=np.float32)}
    except TimeoutException:
        print(f"Timeout simulate() took too long.")
        return {"data": np.full(grid_size, np.nan, dtype=np.float32)}
    except Exception as e:
        print(f"simulate() error: {e}")
        return {"data": np.full(grid_size, np.nan, dtype=np.float32)} 
    
    
def distance(grid1, grid2, threshold=0.0):
    grid1 = grid1["data"]
    grid2 = grid2["data"]
    
    mask1 = (grid1 > threshold).astype(int)
    mask2 = (grid2 > threshold).astype(int)

    jaccard = jaccard_score(mask1.flatten(), mask2.flatten())
    d = 1-jaccard
        
    csv_path = os.path.abspath(f'runs/distance_{d}run.csv')
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(grid1)
        
    print(d)
    return d


acceptor = pyabc.UniformAcceptor()
acceptor.unique_only = False
epsilon_schedule = pyabc.QuantileEpsilon(alpha= 0.6, quantile_multiplier=1)
sampler = pyabc.sampler.MulticoreEvalParallelSampler()
abc_cont = pyabc.ABCSMC(model, prior, distance, population_size=5 ,eps = epsilon_schedule ,sampler=sampler, acceptor=acceptor)

abc_cont.load("sqlite:///" + db,1)
db_path = os.path.abspath(db)


history = abc_cont.run(minimum_epsilon=0.015, max_nr_populations=10, max_total_nr_simulations= 5000)


