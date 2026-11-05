# -*- coding: utf-8 -*-
"""
Service Time Scenario Generation using Smoothed Histogram Sampler

Strategy (from ScenarioGeneration_Complete.ipynb):
1. Load empirical service times from cleaned_data.csv grouped by CumVisits (visit buckets)
2. For each visit bucket (1, 2, 3, 3+), build a smoothed histogram sampler
3. Generate scenarios directly from the histogram sampler
4. Reproducible via seed=132, but different for each patient (independent per-patient sampling)
5. Different scenarios for each day (sample separately for each visit day)

"""


import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm





class create_scenario():
    
    def __init__(self,instance,nb_scenario,distribution,Travel_cov,simulation_type,simulation_mean_learn=None,simulation_std_learn=None):
        """
        Initialize scenario generator using smoothed histogram sampler.
        
        Parameters:
        - instance: Instance object
        - nb_scenario: Number of scenarios
        - distribution: 'normal', 'shifted-gamma', 'shifted-expo'
        - Travel_cov: Travel time coefficient of variation
        """
        self.number_aide = instance.number_aide
        self.number_node = instance.number_node
        self.instance = instance
        travel_matrix = instance.distance_matrix
        number_node = instance.number_node
        self.distribution = distribution
        self.rng = np.random.default_rng(seed=132)
        self.negative_service_time_draws = 0
        self.negative_truncated_service_time_draws = 0
        self.below_lower_truncated_service_time_draws = 0
        

        self.number_scenario=nb_scenario
        

        #Generate travel times per distribution if Travel_cov>0

        if Travel_cov>0:
            
        
            if distribution=="normal":
                
                self.travel_time={s:[0 if i==j else self.normal_sample(travel_matrix[i][j],Travel_cov*travel_matrix[i][j])
                                     for i in range(number_node)for j in range(number_node)] for s in range(nb_scenario)}
                
            elif distribution=="shifted-gamma":
                self.travel_time={s:[0 if i==j else self.gamma_sample(travel_matrix[i][j],Travel_cov)
                                 for i in range(number_node)for j in range(number_node)] for s in range(nb_scenario)}
            
            elif distribution=="shifted-expo":
                self.travel_time={s:[0 if i==j else self.expo_sample(travel_matrix[i][j],Travel_cov)
                                 for i in range(number_node)for j in range(number_node)] for s in range(nb_scenario)}
           

            
        
        else:

            self.travel_time={s:[0 if i==j else travel_matrix[i][j]
                                     for i in range(number_node)for j in range(number_node)] for s in range(nb_scenario)}



        if simulation_type=='empirical-trimmed':
            print(f"service_type {simulation_type}")
            self.service_time = self.build_empirical_service_time('trimmed')
        elif simulation_type=='empirical':
            self.service_time = self.build_empirical_service_time('cleaned')
        elif simulation_type=='normal':
            self.service_time=self.generate_normal_service_time(simulation_mean_learn,simulation_std_learn)
            print(f"Number of negative service time draws: {self.negative_service_time_draws} out of {self.number_scenario * self.instance.number_patient * self.instance.horizon}")
        elif simulation_type=='truncated-normal':
            self.service_time=self.generate_truncated_normal_service_time(simulation_mean_learn,simulation_std_learn,lower_min=30)
            print(f"Number of below lower bound truncated service time draws: {self.below_lower_truncated_service_time_draws} out of {self.number_scenario * self.instance.number_patient * self.instance.horizon}")

    def duration(self,i):
        if i<self.number_aide or i>=self.number_node-self.number_aide:
            return 0
        else:
            return self.instance.list_patient[i-self.number_aide].visit_duration
    
    def normal_sample(self,mean,std):
        
        if mean==0:
            return 0
        else:
            
            x=np.random.normal(mean,std)
            
            while (x<0):
                x=np.random.normal(mean,std)
                
        
        return(x)

    def generate_normal_service_time(self,simulation_mean_learn,simulation_std_learn):

        service_time={}
        negative_count = 0

        service_time = {}
        for p in range(self.instance.number_patient):
            node_id = p + self.instance.number_aide
            service_time[node_id] = {}
            
            for day in range(1, self.instance.horizon + 1):
                mean = simulation_mean_learn[day-1]
                std = simulation_std_learn[day-1]
                scenarios_for_day = self.rng.normal(mean, std, size=self.number_scenario)
                negative_count += int(np.sum(scenarios_for_day < 0))
                scenarios_for_day = np.maximum(scenarios_for_day, 0)
                service_time[node_id][day] = scenarios_for_day
                
        self.negative_service_time_draws = negative_count

        return service_time
    def generate_truncated_normal_service_time(self,simulation_mean_learn,simulation_std_learn,lower_min=30 ):

        service_time = {}
        below_lower_count = 0
        for p in range(self.instance.number_patient):
            node_id = p + self.instance.number_aide
            service_time[node_id] = {}
            
            for day in range(1, self.instance.horizon + 1):
                mean = simulation_mean_learn[day-1]
                std = simulation_std_learn[day-1]
                samples = self.rng.normal(mean, std, size=self.number_scenario)
                below_lower_count += int(np.sum(samples < lower_min))
                scenarios_for_day = np.maximum(samples, lower_min)
                service_time[node_id][day] = scenarios_for_day
                
        self.below_lower_truncated_service_time_draws = below_lower_count

        return service_time

    def normal_sample_truncated(self,mean,std,type_service):

        lower=0.5*type_service
        upper=2*type_service


        a= (lower - mean) / std
        b= (upper - mean) / std

        samples=truncnorm(a=a, b=b, loc= mean, scale=std).rvs(size=self.number_scenario, random_state=self.rng)

        return samples

    
    def gamma_sample(self,mean,cov):
        if mean==0:
            return 0
        else:
            theta=(cov*mean)/2  #the scale parameter of gamma when shape, k, equals to 4 [std(x)=sqrt(shape*theta^2)=mean*cov]
            shifted_mean= (2*cov-1)*mean   # [E(x)=shape*theta] we substance the shifted_mean to finally have the proper mean
            x=np.random.gamma(shape=4,scale=theta)-shifted_mean
            
            while (x<0):
                x=np.random.gamma(shape=4,scale=theta)-shifted_mean
                
        return(x) 
    
    def expo_sample(self,mean,cov):
        if mean==0:
            return 0
        else: 
            rate= 1/(mean*cov)  # lambda or rate at exponential distribution! 
            beta= 1/rate # scale parameter equals to 1\lambda, this equals the std and mean of the distribution 
            shifted_mean= (cov-1)*mean 
            x=np.random.exponential(scale=beta)-shifted_mean
            
            while (x<0):
                x=np.random.exponential(scale=beta)-shifted_mean
                
        #if x>0 else self.normal_sample(mean,std)
    

    def build_empirical_service_time(self, type):
        """
        Generate service times using smoothed histogram sampler per visit bucket.
        
        Structure: service_time[patient_node][visit_day] = array of scenarios
        Each patient gets independent samples from the appropriate visit bucket.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if type=='cleaned':
            csv_path = os.path.join(os.path.dirname(base_dir), 'data-analysis', 'cleaned_data.csv')
        elif type=='trimmed':
            csv_path = os.path.join(os.path.dirname(base_dir), 'data-analysis', 'trimmed_data_no-outliers.csv')
        

        # Load empirical data
        df = pd.read_csv(csv_path, usecols=['ActualServiceTime', 'CumVisits'])
        
        #trim data 
        #df=self.trim_data (df)

        # Group by visit bucket and build samplers
        visit_buckets = dict(list(df.groupby('CumVisits')))  #returns a dict with values '1','2','3','3+' and corresponding dataframes
        samplers_dict = {}

        bucket_seed_offset = {'1': 0, '2': 1, '3': 2, '3+': 3}

        for bucket, bucket_data in visit_buckets.items():
            x = bucket_data['ActualServiceTime'].to_numpy()
            if len(x) >= 5:
                # Deterministic seed per bucket to keep runs reproducible
                seed = 132 + bucket_seed_offset.get(str(bucket), 10)
                sampler, _, _ = self.build_hist_sampler(x, seed=seed)
                samplers_dict[bucket] = sampler

        # Generate service times: per patient, per visit day, all scenarios
        service_time = {}
        for p in range(self.instance.number_patient):
            node_id = p + self.instance.number_aide
            service_time[node_id] = {}
            
            for day in range(1, self.instance.horizon + 1):
                bucket = self.bucket_for_day(day)
                
                if bucket in samplers_dict:
                    # Generate scenarios for this patient/day from histogram sampler
                    sampler = samplers_dict[bucket]
                    scenarios_for_day = sampler(self.number_scenario, integer=True)
                    service_time[node_id][day] = scenarios_for_day
                

        return service_time

    def trim_data(self,df):
        group= df.groupby('CumVisits').agg(
            outlier=('ActualServiceTime', lambda x: np.percentile(x, 75) + 1.5 * (np.percentile(x, 75) - np.percentile(x, 25)))
            ).reset_index()
        data_with_outlier = df.merge(group, on='CumVisits', how='left')

        data_trimmed = data_with_outlier[data_with_outlier['ActualServiceTime'] < data_with_outlier['outlier']].drop(columns=['outlier'])
        return data_trimmed
        


    
    def bucket_for_day(self, day_index):
        """Map day index to visit bucket."""
        return str(day_index) if day_index <= 3 else '3+'

    def build_hist_sampler(self, x, lo=30, hi=180, bin_width=5, lam=0.5, seed=42):
        """
        Build smoothed histogram sampler using Laplace smoothing.
        P(bin) = (count(bin) + lambda) / (total + lambda * n_bins)
        """
        rng = np.random.default_rng(seed)
        
        # Create bins
        edges = np.arange(lo, hi + bin_width, bin_width)
        counts, _ = np.histogram(x, bins=edges)
        
        B = len(counts)
        probs = (counts + lam) / (counts.sum() + lam * B)
        
        lows = edges[:-1]
        highs = edges[1:]
        
        def sample(n, integer=True):
            """Generate n scenarios from smoothed histogram."""
            b = rng.choice(B, size=n, p=probs)
            y = rng.uniform(lows[b], highs[b])
            if integer:
                y = np.rint(y)
            y = np.clip(y, lo, hi)
            return y.astype(int if integer else float)
        
        return sample, probs, edges

