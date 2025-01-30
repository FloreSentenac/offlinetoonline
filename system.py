import numpy as np

class System:
    def __init__(self, K, means, off_data):
        self.K = K
        self.means = means
        self.mustar=np.max(means)
        self.off_data = off_data
        self.sample_count = np.array([np.size(x) for x in off_data])
        self.sum_rewards=[np.sum(x) for x in off_data]
        self.m=np.sum(self.sample_count)
        self.mu0=np.sum(self.sample_count*self.means)/self.m
        self.hatmu0= np.sum(self.sum_rewards)/self.m
        self.history=[0]
        self.t=0


    def reinit(self):
        self.sample_count = np.array([np.size(x) for x in self.off_data])
        self.sum_rewards=[np.sum(x) for x in self.off_data]
        self.m=np.sum(self.sample_count)
        self.history=[0]
        self.t=0

    def round(self,algo,t,reward_table=None):
        a = algo.choice(self,t)
        self.sample_count[a]+=1
        if reward_table is None:
            self.sum_rewards[a]+=np.random.binomial(1, self.means[a])
            self.history.append(self.history[-1]+self.mustar-self.means[a])
        else:
            row = np.random.randint(0, 99999)
            self.sum_rewards[a]+=reward_table[row][a]
            self.history.append(self.history[-1]+reward_table[row][a])

    def run(self,T, algo,reward_table=None):
        for t in range(T):
            self.round(algo,t,reward_table)

    def multiple_run(self,T,algo,n=1, version=None,reward_table=None):
        hist=[]
        sample_count=[]
        for a in range(n):
            self.reinit()
            if algo.name=="OtO":
                algo.init_OtO(self,version)
            self.run(T, algo,reward_table)
            hist.append(self.history)
            sample_count.append(self.sample_count)
        hist = np.mean(hist,axis=0)
        return hist
