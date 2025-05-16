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
pp_mu = 0.0001
pp_sigma = 0.001
pp_lower, pp_upper = 0, 1

# Standardized bounds
pp_a, pp_b = (pp_lower - pp_mu) / pp_sigma, (pp_upper - pp_mu) / pp_sigma

# Parameters
wm_mu = 0.075
wm_sigma = 0.02
wm_lower, wm_upper = 0, 1

# Standardized bounds
wm_a, wm_b = (wm_lower - wm_mu) / wm_sigma, (wm_upper - wm_mu) / wm_sigma

# Parameters
gm_mu = 0.8
gm_sigma = 0.1
gm_lower, gm_upper = 0, 1

# Standardized bounds
gm_a, gm_b = (gm_lower - gm_mu) / gm_sigma, (gm_upper - gm_mu) / gm_sigma

# Parameters
m_mu = 0.6
m_sigma = 0.1
m_lower, m_upper = 0, 1

# Standardized bounds
m_a, m_b = (m_lower - m_mu) / m_sigma, (m_upper - m_mu) / m_sigma

# Parameters
theta_mp_mu = 6
theta_mp_sigma = 1
theta_mp_lower, theta_mp_upper = 0, 10

# Standardized bounds
theta_mp_a, theta_mp_b = (theta_mp_lower - theta_mp_mu) / theta_mp_sigma, (theta_mp_upper - theta_mp_mu) / theta_mp_sigma


# Parameters
theta_pm_mu = 8
theta_pm_sigma = 1
theta_pm_lower, theta_pm_upper = 0, 10

# Standardized bounds
theta_pm_a, theta_pm_b = (theta_pm_lower - theta_pm_mu) / theta_pm_sigma, (theta_pm_upper - theta_pm_mu) / theta_pm_sigma


# Parameters
alpha_mu = 10
alpha_sigma = 1
alpha_lower, alpha_upper = 6,10 

# Standardized bounds
alpha_a, alpha_b = (alpha_lower - alpha_mu) / alpha_sigma, (alpha_upper - alpha_mu) / alpha_sigma

# Parameters
a_pm_mu = 0.2
a_pm_sigma = 0.05
a_pm_lower, a_pm_upper = 0, 1

# Standardized bounds
a_pm_a, a_pm_b = (a_pm_lower - a_pm_mu) / a_pm_sigma, (a_pm_upper - a_pm_mu) / a_pm_sigma

# Parameters
a_mp_mu = 0.1
a_mp_sigma = 0.05
a_mp_lower, a_mp_upper = 0, 1

# Standardized bounds
a_mp_a, a_mp_b = (a_mp_lower - a_mp_mu) / a_mp_sigma, (a_mp_upper - a_mp_mu) / a_mp_sigma

# Parameters
b_pm_mu = 0.4
b_pm_sigma = 0.05
b_pm_lower, b_pm_upper = 0, 1

# Standardized bounds
b_pm_a, b_pm_b = (b_pm_lower - b_pm_mu) / b_pm_sigma, (b_pm_upper - b_pm_mu) / b_pm_sigma

# Parameters
b_mp_mu = 0.2
b_mp_sigma = 0.03
b_mp_lower, b_mp_upper = 0, 1

# Standardized bounds
b_mp_a, b_mp_b = (b_mp_lower - b_mp_mu) / b_mp_sigma, (b_mp_upper - b_mp_mu) / b_mp_sigma

# Parameters
k_pm_mu = 0.4
k_pm_sigma = 0.05
k_pm_lower, k_pm_upper = 0, 10

# Standardized bounds
k_pm_a, k_pm_b = (k_pm_lower - k_pm_mu) / k_pm_sigma, (k_pm_upper - k_pm_mu) / k_pm_sigma

# Parameters
k_mp_mu = 0.4
k_mp_sigma = 0.05
k_mp_lower, k_mp_upper = 0, 10

# Standardized bounds
k_mp_a, k_mp_b = (k_mp_lower - k_mp_mu) / k_mp_sigma, (k_mp_upper - k_mp_mu) / k_mp_sigma


#Create initial cell grid based on mr_data
matrix = np.loadtxt(start_intensity,delimiter=',')
cells = create_initial_tumor(matrix,eff_radie, mid_point)
grid_size = matrix.shape
observation = np.loadtxt(end_intensity,delimiter=',')


prior = pyabc.Distribution(pp=pyabc.RV("truncnorm",pp_a, pp_b, loc=pp_mu, scale=pp_sigma),
                           m = pyabc.RV("truncnorm", m_a, m_b, loc=m_mu, scale=m_sigma),
                           alpha=pyabc.RV("truncnorm",alpha_a, alpha_b, loc=alpha_mu, scale=alpha_sigma),
                           a_pm=pyabc.RV("truncnorm",a_pm_a, a_pm_b, loc=a_pm_mu, scale=a_pm_sigma),
                           b_pm= pyabc.RV("truncnorm",b_pm_a, b_pm_b, loc = b_pm_mu, scale=b_pm_sigma),
                           theta_pm=pyabc.RV("truncnorm",theta_pm_a, theta_pm_b, loc=theta_pm_mu, scale=theta_pm_sigma),
                           k_pm=pyabc.RV("truncnorm",k_pm_a, k_pm_b, loc=k_pm_mu, scale=k_pm_sigma),
                           a_mp=pyabc.RV("truncnorm",a_mp_a, a_mp_b, loc=a_mp_mu, scale=a_mp_sigma),
                           b_mp=pyabc.RV("truncnorm",b_mp_a, b_mp_b, loc = b_mp_mu, scale=b_mp_sigma),
                           theta_mp=pyabc.RV("truncnorm",theta_mp_a, theta_mp_b, loc=theta_mp_mu, scale=theta_mp_sigma),
                           k_mp=pyabc.RV("truncnorm",k_pm_a, k_pm_b, loc=k_pm_mu, scale=k_pm_sigma),
)



def model(parameter):
    try:
        signal.alarm(3600)  # timeout in 1 hour
        result = simulate(
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
acceptor.unique_only = True
epsilon_schedule = pyabc.QuantileEpsilon(initial_epsilon=0.7, alpha = 0.6, quantile_multiplier=1)
sampler = pyabc.sampler.MulticoreEvalParallelSampler()
abc = pyabc.ABCSMC(model, prior, distance, population_size=100 ,eps = epsilon_schedule ,sampler=sampler,acceptor=acceptor)

db_path = os.path.abspath(db)

#Database
abc.new("sqlite:///" + db_path, {"data": observation})


history = abc.run(minimum_epsilon=0.015, max_nr_populations=10, max_total_nr_simulations= 5000)



