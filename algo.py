import numpy as np

class Algo:
    def __init__(self, name, delta,K,alpha=None,T=None,version=None):
        self.name=name
        self.delta=delta
        self.alpha=alpha
        self.T=T
        self.version=version
        if version=="unknown horizon":
            self.T=2
        self.lower_bound=np.zeros(K)
        self.ref_arm=None


    def update_upper_bound(self,system,t):
        #print("here with",self.name)
        self.upper_bound = system.sum_rewards/system.sample_count
        if self.version=="unknown horizon":
            self.upper_bound+=np.sqrt(np.log(system.K*(t+1)**2/self.delta)/system.sample_count)
        else:

            self.upper_bound+=np.sqrt(np.log(system.K/self.delta)/system.sample_count)

    def update_lower_bound(self,system,t):
        lower_bound = system.sum_rewards/system.sample_count
        if self.version=="unknown horizon":
            lower_bound-=np.sqrt(np.log(system.K*(t+1)**2/self.delta)/system.sample_count)
        else:
            lower_bound-=np.sqrt(np.log(system.K/self.delta)/system.sample_count)
        lower_bound=np.nan_to_num(lower_bound)
        self.lower_bound=np.maximum(self.lower_bound,lower_bound)

    def choice(self,system,t):
        self.update_upper_bound(system,t)
        self.update_lower_bound(system,t)
        if self.name=="UCB":
            return np.argmax(self.upper_bound)
        if self.name=="LCB":
            return np.argmax(self.lower_bound)
        if self.name=="OtO":
            a= np.argmax(self.upper_bound)
            if self.check_buget(system,a):
                a= np.argmax(self.upper_bound)
                self.arm_count[a]+=1
            else:
                a= np.argmax(self.lower_bound)
                self.lcb_count+=1
                self.arm_count_lcb[a]+=1
                if self.version=="partial update param" or self.version=="full update param":
                    self.arm_count[a]+=1
            self.update_budget(system,a,t)
            return a


    def init_OtO(self,system,version):
        self.lower_bound=np.zeros(system.K)
        self.update_lower_bound(system,0)
        self.lcb_count=0
        self.arm_count=np.zeros(system.K)
        self.arm_count_lcb=np.zeros(system.K)
        self.lcb=[]
        self.version=version
        if version=="unknown horizon":
            self.T=2
        if version=="full update param":
            self.ref_arm=np.argmax(self.lower_bound)
        self.init_budget(system)


    def init_budget(self,system):
        self.beta= np.sum(np.sqrt(system.sample_count))/np.sqrt(system.m)*np.sqrt(np.log(1/self.delta)/2)
        self.gamma=np.max(self.lower_bound)-self.alpha*self.beta/np.sqrt(system.m)
        self.future_budget=(self.T-1)*self.alpha*self.beta/np.sqrt(system.m)
        self.budget=self.future_budget


    def check_buget(self,system,a):
        return self.budget+self.lower_bound[a]-self.gamma>0

    def update_budget(self,system,a,t):
        if self.T<t:
            self.future_budget+=self.T*self.alpha*self.beta/np.sqrt(system.m)
            self.T=2*self.T
        self.future_budget-=self.alpha*self.beta/np.sqrt(system.m)
        if self.version=="main" or self.version=="unknown horizon":
            self.budget=np.sum(self.arm_count*(self.lower_bound-self.gamma))+self.future_budget
            self.budget+= self.lcb_count*self.alpha*self.beta/np.sqrt(system.m)
        if self.version=="partial update param":
            self.budget=np.sum(self.arm_count*(self.lower_bound-self.gamma))+self.future_budget
        if self.version=="full update param":
            self.budget=np.sum(self.arm_count*(self.lower_bound-self.lower_bound[self.ref_arm]+self.alpha*self.beta/np.sqrt(system.m)))+self.future_budget

