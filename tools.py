import numpy as np
from algo import Algo
from system import System
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

def repeat_exp_debug(K, m, T, means, cutoff, version, delta=None,use_reward=False, reward_table=None, alpha=0.25, n_exp=200):
    if delta==None:
        delta =1 / T ** 2
    UCB = Algo("UCB", delta, K,version=version)
    OtO = Algo("OtO", delta, K, alpha, T, version=version)
    LCB = Algo("LCB", delta, K, 1,version=version)
    # Generate off-policy data based on the cutoff
    off_data = [np.random.binomial(1, means[i], int(2 * m / K * (i < cutoff))) for i in range(K)]
    system = System(K, means, off_data)
    print(UCB.name,"at init", UCB.version)
    # Run the algorithms and collect results
    result_UCB = system.multiple_run(T, UCB, 1, version=version)
    result_OtO = system.multiple_run(T, OtO, 1, version=version)
    result_LCB = system.multiple_run(T, LCB, 1, version=version)

    return result_UCB, result_OtO, result_LCB

def repeat_exp(K, m, T, means, cutoff, version, delta=None,use_reward=False, reward_table=None, alpha=0.25, n_exp=200):
    if delta==None:
        delta =1 / T ** 2
    UCB = Algo("UCB", delta, K,version=version)
    OtO = Algo("OtO", delta, K, alpha, T, version=version)
    LCB = Algo("LCB", delta, K, 1,version=version)
    def run_experiment():
        # Generate off-policy data based on the cutoff
        off_data = [np.random.binomial(1, means[i], int(2 * m / K * (i < cutoff))) for i in range(K)]
        system = System(K, means, off_data)
        
        # Run the algorithms and collect results
        result_UCB = system.multiple_run(T, UCB, 1, version=version)
        result_OtO = system.multiple_run(T, OtO, 1, version=version)
        result_LCB = system.multiple_run(T, LCB, 1, version=version)
        
        return result_UCB, result_OtO, result_LCB

    # Run in parallel and use tqdm for tracking progress
    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)() for _ in tqdm(range(n_exp), total=n_exp)
    )

    # Unpack the results for each algorithm into separate lists
    hist_UCB, hist_OtO, hist_LCB = zip(*results)
    return list(hist_UCB), list(hist_OtO), list(hist_LCB)


def repeat_exp_real_data(K, m, T, means, cutoff, reward_table,offline_data,delta=None,version="main", alpha=1.,n_exp=200):
    if delta==None:
        delta =1 / T ** 2
    UCB = Algo("UCB", delta, K)
    OtO = Algo("OtO", delta, K, alpha=alpha, T=T)
    LCB = Algo("LCB", delta, K)

    def run_iteration():
        off_data = [offline_data]
        for i in range(1, K):
            off_data.append([])

        system = System(K, means, off_data)
        UCB_instance = Algo("UCB", delta, K)
        result_UCB = system.multiple_run(T, UCB_instance, 1, reward_table=reward_table)
        result_OtO = system.multiple_run(T, OtO, 1, version, reward_table=reward_table)
        result_LCB = system.multiple_run(T, LCB, 1, reward_table=reward_table)
        return result_UCB, result_OtO, result_LCB

    # Run in parallel and collect results
    results = Parallel(n_jobs=-1)(delayed(run_iteration)() for _ in tqdm(range(n_exp)))

    # Separate the results for each algorithm
    hist_UCB, hist_OtO, hist_LCB = zip(*results)
    return list(hist_UCB), list(hist_OtO), list(hist_LCB)

def repeat_exp_real_data_oto(K, m, T, means, cutoff, reward_table,offline_data,version="main" ,delta=None,alpha=1.,n_exp=200):
    if delta==None:
        delta =1 / T ** 2
    OtO = Algo("OtO", delta, K, alpha=alpha, T=T)

    def run_iteration():
       # indices = np.random.randint(0, len(reward_table), size=m)
        off_data = [offline_data]#[reward_table[indices, 0]]
        for i in range(1, K):
            off_data.append([])
        system = System(K, means, off_data)
        result_OtO = system.multiple_run(T, OtO, 1,version, reward_table=reward_table)
        return  result_OtO

    # Run in parallel and collect results
    results = Parallel(n_jobs=-1)(delayed(run_iteration)() for _ in tqdm(range(n_exp)))

    # Separate the results for each algorithm
    #hist_OtO = zip(*results)
    return list([*results])



def plot_with_trajectory_confidence_interval(hist_list, label, color):
  hist_array = np.array(hist_list)  # Convert to NumPy array
  
  # Plot median regret
  mean_regret = np.mean(hist_array, axis=0)
  plt.plot(mean_regret, color=color, label=label)  # Plot mean regret
  
  # Calculate the std
  std_regret = np.std(hist_array, axis=0)
  lower_bound = mean_regret-2*std_regret  
  upper_bound = mean_regret+2*std_regret  

  # Fill confidence interval
  plt.fill_between(range(len(mean_regret)), lower_bound, upper_bound, color=color, alpha=0.2)