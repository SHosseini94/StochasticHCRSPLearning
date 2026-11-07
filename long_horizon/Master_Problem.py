# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:47:08 2024

@author: saeed
"""



#from xml.parsers.expat import model
from gurobipy import *
import os
import time
import sys
import numpy as np
import random
from itertools import product
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from Subproblem import Subproblem
from Subproblem import No_Learn_Subproblem
from Subproblem import Deterministic_No_Learn_Subproblem

logger = logging.getLogger(__name__)
#from Subproblem_fixedVisitTime import Subproblem   # if we want to fix visit time for existing patient 

def lazy_callback(model,where):

    if where==GRB.Callback.MIPSOL:

        model._call+=1

        # Extract solution values — must happen in the main callback thread
        yval=model.cbGetSolution(model._yvars)
        numval=model.cbGetSolution(model._numvars)
        freqval=model.cbGetSolution(model._freqvars)

        list_Pik={k.id:{d:[i for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5] for d in
                                 range(model._instance.horizon)} for k in model._activeAides}

        list_visit={k.id:{d:{i:int(np.round(numval[kk,i,dd])) for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5} for d in
                                 range(model._instance.horizon)} for k in model._activeAides}

        list_Pik_frequency={k.id:{d:{i:p for kk,i,dd,p in model._freqAssignment.select(k.id,'*','*','*') if dd==d and freqval[kk,i,dd,p]>0.5} for d in
                                 range(model._instance.horizon)} for k in model._activeAides}

        Solve={k.id:0 for k in model._activeAides}
        for k in Solve:
            for d in range(model._instance.horizon):
                if len(list_Pik[k][d])>0:
                    Solve[k]=1
                    break

        # Build list of aides that need a subproblem solve (not already cached as feasible)
        aides_to_solve=[]
        for k in model._activeAides:
            if Solve[k.id]==0:
                continue
            value=""
            for d in range(model._instance.horizon):
                if list_Pik[k.id][d]:
                    for b in list_Pik[k.id][d]:
                        value+=str(b)+"({})".format(list_Pik_frequency[k.id][d][b])+'-'
                    value+="/"
            if value not in model._instance.list_feasible_per_aide[k.id]:
                aides_to_solve.append(k)
                logger.info("----------------------------------")
                logger.info("Solve Aide %s", k.id)
                logger.info("list_Pik: %s", list_Pik[k.id])
                logger.info("list_visit: %s", list_visit[k.id])
                logger.info("list_Pik_frequency: %s", list_Pik_frequency[k.id])

        if not aides_to_solve:
            return

        # Worker: solve one subproblem and return results (no Gurobi API calls here)
        def solve_one(k):
            subproblem=Subproblem(model._instance,k.id,list_Pik[k.id],list_visit[k.id],
                                  list_Pik_frequency[k.id],model._overtime,model._alpha,
                                  model._beta,model._TCov,model._meanLearning,model._stdLearning)
            t0=time.time()
            over_time_status,status,_,_,complete_sub,dic_cut=subproblem.generate_cut(False)
            elapsed=time.time()-t0
            return k.id, status, complete_sub, dic_cut, elapsed

        # Solve all pending subproblems in parallel — CP Optimizer releases the GIL
        # so threads give true parallelism.  cbLazy must be called in the main thread,
        # so we collect all results first and then add cuts below.
        n_workers=min(len(aides_to_solve), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures={executor.submit(solve_one, k): k for k in aides_to_solve}
            for future in as_completed(futures):
                kid, status, complete_sub, dic_cut, elapsed=future.result()
                model._subcall+=1
                model._subTime+=elapsed
                model._complete_sub+=complete_sub

                if status==False:
                    model._inf_complete_sub+=1
                    for d in dic_cut:
                        if len(dic_cut[d])>0:
                            model.cbLazy(quicksum((1-model._freqvars[kid,i,d,p]) for (i,p) in dic_cut[d])>=1)
                            model._nblazy+=1
                    logger.info("cut list aide %s: %s", kid, dic_cut)
                    logger.info("-----------------------------------")

                    
                
    


class Master_Problem():  # with learning and only time window relaxation

    def __init__(self,file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov, mean_learn,std_learn,team,team_size,similar_first,past_aide_visits=None,prev_day_assignment=None):

        #print("Master Problem is created")

        self.instance=instance
        self.overtime=overtime
        self.alpha=alpha
        self.beta=beta
        self.gamma1=gamma1
        self.gamma2=gamma2
        self.TCov=TCov
        self.mean_learning=mean_learn
        self.std_learning=std_learn
        self.team_size=team_size
        self.similar_first=similar_first
        self.file_name=file_name
        # past_aide_visits: {patient_id: {aide_id: visit_count_in_previous_run}}
        # Patients in this dict are "returning" and will have delta fixed to 1.
        self.past_aide_visits = past_aide_visits if past_aide_visits is not None else {}
        # prev_day_assignment: {patient_id: {day: aide_id}} — exact solution from last run.
        # Used as warm start for returning patients.
        self.prev_day_assignment = prev_day_assignment if prev_day_assignment is not None else {}
        # result_day_assignment is populated after solving; pass to next run as prev_day_assignment.
        self.result_day_assignment = {}
        
        
        
        
        
        #new patient are those who are available to be assigned to an aide
        self.new_patient=[instance.list_patient[i] for i in range(instance.number_patient) if instance.list_patient[i].
                            availability==2]
        
        
        #active aide are those who are available to serve patients
        self.active_aide=[instance.list_aide[i] for i in range(instance.number_aide) if instance.list_aide[i].
                            availability==2]
        
        self.list_aide_active_id=[]
        for k in self.active_aide:
            self.list_aide_active_id.append(k.id)

        self.list_patient_new_id=[]
        for i in self.new_patient:
            self.list_patient_new_id.append(i.id)

        #self.nb_patient=len(self.new_patient) 
        #self.nb_aide=len(self.active_aide)  
        

        self.horizon=instance.horizon
        
        if team:
            self.team_maker()
            
            #import pdb; pdb.set_trace()

            self.day_assignment=tuplelist([(k,i,d) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1])

            self.day_visit_frequency=tuplelist([(k,i,d,p) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i] for p in self.patient_visit_frequency[i][d][k]])


        else:
            self.day_assignment=tuplelist([(k.id,i.id,d) for k in self.active_aide for i in self.new_patient for d in range(self.horizon) if i.pre_assigned_day[d]==1])
        
        #to calculate maximum number of potential visits for each (k,i,d)
        
        self.num_day_assignment=[]
        for k,i,d in self.day_assignment:
            max_num_visit=sum([self.instance.list_patient[i].pre_assigned_day[dd] for dd in range(d+1)])
            for p in range(max_num_visit):
                self.num_day_assignment.append((k,i,d,p))

        
        self.learned_visit_duration={}
        self.non_learned_visit_duration={}
        for i in self.new_patient:
            p=0
            for d in range(self.horizon):
                if i.pre_assigned_day[d]==1:
                    self.learned_visit_duration[i.id,d]=int(np.round(i.visit_duration*self.mean_learning[p]))
                    self.non_learned_visit_duration[i.id,d]=i.visit_duration
                    p+=1
                
        self.minimum_mean_service_time=sum(self.learned_visit_duration.values())
        self.maximum_mean_service_time=sum(self.non_learned_visit_duration.values())

       
        self.minimum_travel_time=0
        self.maximum_travel_time=0          
        
        #self.new_assignment=tuplelist([(k.id,i.id) for k in self.active_aide for i in self.free_patient])
        #print("the potential patient-aide assignment") 
        #print(self.new_assignment)
        #self.all_assignment=self.current_assignment+self.new_assignment
        #print("all assignments:")
        #print(self.all_assignment)
        
        self.morning_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[1]<=720]
        self.evening_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[0]>=720]

        logger.info("morning patients: %s", self.morning_patient)
        logger.info("evening patients: %s", self.evening_patient)

        #self.ugmentedDurationFrom={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_from[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}
        
        #self.ugmentedDurationTo={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_to[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}


        
        self.ugmentedDurationFrom={(k,i,d,p):self.patient_duration_frequency[i][p]+self.min_travel_from[i][d][k] for k,i,d,p in self.day_visit_frequency}

        self.ugmentedDurationTo={(k,i,d,p):self.patient_duration_frequency[i][p]+self.min_travel_to[i][d][k] for k,i,d,p in self.day_visit_frequency}

        
                    
                            
    
    """ It is a function to call the subproblem
        when the master problem finds a feasible solution
        and add feasibility cuts when they are required.
        It links the master and subproblem.
    """

    @staticmethod
    def extract_past_visits(instance):
        """Extract aide_assigned from a solved instance for use in the next run.

        Returns {patient_id: {aide_id: visit_count}} for accepted patients.
        Pass the result as past_aide_visits= to the next Master_Problem constructor.
        """
        past={}
        for pat in instance.list_patient:
            if pat.availability==2 and pat.aide_assigned:
                past[pat.id]=dict(pat.aide_assigned)
        return past

    def team_maker(self):
        random.seed(42)  # Set the seed once for reproducibility
        #self.rng=np.random.default_rng(seed=2025)

        self.list_assignment={}
        for i in self.list_patient_new_id:

            self.list_assignment[i]=[]

            potential_aide=[]

            earliest_start=self.instance.list_patient[i].time_window[0]
            latest_start=self.instance.list_patient[i].time_window[1]-self.instance.list_patient[i].visit_duration
            visit_duration=self.instance.list_patient[i].visit_duration


            for k in self.list_aide_active_id:
                travel_time=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                time_window_start=self.instance.list_aide[k].time_window_start[0]
                time_window_end=self.instance.list_aide[k].time_window_end[1]

                if time_window_start + travel_time > latest_start:  # cannot reach the patient before his/her latest start time
                    continue
                elif travel_time+visit_duration > time_window_end + self.overtime - earliest_start:
                    continue
                elif earliest_start + visit_duration + self.beta * (0.25 * visit_duration) > time_window_end + self.overtime:
                    continue
                else:
                    potential_aide.append(k)

            if self.past_aide_visits.get(i):
                # Returning patient: past aides must be in the team (if still feasible)
                past_aides=[k for k in self.past_aide_visits[i] if k in potential_aide]
                remaining=[k for k in potential_aide if k not in past_aides]
                slots=max(0,self.team_size-len(past_aides))
                if len(remaining)<=slots:
                    self.list_assignment[i]=past_aides+remaining
                else:
                    self.list_assignment[i]=past_aides+random.sample(remaining,slots)
            elif len(potential_aide)<=self.team_size:
                self.list_assignment[i]=potential_aide
            else:
                selected_aide=random.sample(potential_aide, self.team_size)
                self.list_assignment[i]=selected_aide

        self.patient_pattern={}

        for i in self.list_patient_new_id:
            self.patient_pattern[i]={}
            visit_days=[ d for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1]
            patterns=list(product(self.list_assignment[i], repeat=len(visit_days)))
            self.patient_pattern[i]={n:{day:p[d] for d,day in enumerate(visit_days)} for n,p in enumerate(patterns)}

        self.aide_day_patient={}
        for k in self.list_aide_active_id:
            #location_start=self.instance.list_aide[k].location_start + self.instance.number_node
            #location_end=self.instance.list_aide[k].location_end + self.instance.number_node

            self.aide_day_patient[k]={}
            for d in range(self.horizon):
                self.aide_day_patient[k][d]=[]
                for i in self.list_patient_new_id:
                    if self.instance.list_patient[i].pre_assigned_day[d]==1 and k in self.list_assignment[i]:
                        self.aide_day_patient[k][d].append(i)
                
        self.min_travel_from={}
        self.min_travel_to={}

        for i in self.list_patient_new_id:
            self.min_travel_from[i]={}
            self.min_travel_to[i]={}
            for d in range(self.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    self.min_travel_from[i][d]={}
                    self.min_travel_to[i][d]={}
                    for k in self.list_assignment[i]:
                        minTravelFrom=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]<minTravelFrom:
                                minTravelFrom=self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]
                        self.min_travel_from[i][d][k]=minTravelFrom

                        minTravelTo=100
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]<minTravelTo:
                                minTravelTo=self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]
                        
                        if minTravelTo==100:
                            minTravelTo=0
                        self.min_travel_to[i][d][k]=minTravelTo

        
    
        
        self.patient_pattern_duration={}
        for i in self.list_patient_new_id:
            self.patient_pattern_duration[i]={}
            visit_duration=self.instance.list_patient[i].visit_duration
            for pattern_id,pattern in self.patient_pattern[i].items():
                self.patient_pattern_duration[i][pattern_id]={}
                aide_assignment_count={}
                for day,aide in pattern.items():
                    if aide not in aide_assignment_count:
                        aide_assignment_count[aide]=0
                    
                    aide_assignment_count[aide]+=1
                    learning_index=aide_assignment_count[aide]-1

                    self.patient_pattern_duration[i][pattern_id][day]=int(np.round(visit_duration*self.mean_learning[learning_index]))

        
        self.patient_visit_frequency={}
        self.patient_duration_frequency={}
        self.patient_day_frequency={}
        self.patient_frequency_day={}  # {i: {k: {p: [days]}}} — per aide

        for i in self.list_patient_new_id:
            is_morning=self.instance.list_patient[i].time_window[1]<=720
            self.patient_visit_frequency[i]={}
            self.patient_duration_frequency[i]={}
            self.patient_day_frequency[i]={}
            self.patient_frequency_day[i]={k:{} for k in self.list_assignment[i]}

            total_new_visit=0  # cumulative new visits in this run
            for d in range(self.instance.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    total_new_visit+=1
                    self.patient_visit_frequency[i][d]={}
                    day_p_union=set()

                    for k in self.list_assignment[i]:
                        past_k=self.past_aide_visits.get(i,{}).get(k,0)
                        # Global p values aide k can take on day d: past_k+1 … past_k+total_new_visit
                        p_vals=list(range(past_k+1, past_k+total_new_visit+1))
                        self.patient_visit_frequency[i][d][k]=p_vals
                        day_p_union.update(p_vals)

                        for p in p_vals:
                            # duration uses global visit index p (correct learning factor);
                            # if p exceeds the learning curve, the last entry is used (plateau assumption)
                            if p not in self.patient_duration_frequency[i]:
                                p_idx=min(p-1, len(self.mean_learning)-1)
                                if is_morning:
                                    self.patient_duration_frequency[i][p]=(self.mean_learning[p_idx])+self.gamma1*(self.std_learning[p_idx])
                                else:
                                    self.patient_duration_frequency[i][p]=(self.mean_learning[p_idx])+self.gamma2*(self.std_learning[p_idx])

                            # j-th new visit by aide k (j = p - past_k) can happen on day d
                            # since total_new_visit >= j (all p in p_vals satisfy this)
                            if p not in self.patient_frequency_day[i][k]:
                                self.patient_frequency_day[i][k][p]=[]
                            self.patient_frequency_day[i][k][p].append(d)

                    self.patient_day_frequency[i][d]=list(day_p_union)
        
        """
        print("patient frequency day:")
        for i in self.list_patient_new_id:
            for p in self.patient_frequency_day[i]:
                print("patient {}, frequency {}, days {}".format(i,p,self.patient_frequency_day[i][p]))

        """


                    
        

            
                





            



        
    
    
    def _finalize_solution(self,m,y,num,freq):
        """Extract the current incumbent from m and run the final exact
        subproblem solve (generate_cut(True)) for every aide with a visit,
        to obtain schedules/routes. Valid for any status as long as
        m.SolCount>0 (OPTIMAL, or TIME_LIMIT/INTERRUPTED with a feasible
        incumbent), since .X reads the incumbent regardless of status.
        """

        file_path=os.path.join(self.BASE_DIR, 'result','learn', f"{self.file_name}-Learning-TimeWindowRelaxation-solution-EVT.sol")
        m.write(file_path)

        # Extract per-day assignment for use as warm start in the next run
        self.result_day_assignment={}
        for k,i,d in self.day_assignment:
            if y[k,i,d].X>0.5:
                if i not in self.result_day_assignment:
                    self.result_day_assignment[i]={}
                self.result_day_assignment[i][d]=k

        f={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5] for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}

        list_visit={k:{d:{i:int(np.round(num[kk,i,dd].getAttr('X'))) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5} for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}
        list_Pik_frequency={k:{d:{i:p for kk,i,dd,p in self.day_visit_frequency.select(k,'*','*','*') if dd==d and freq[kk,i,dd,p].getAttr('X')>0.5} for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}

        Solve={k.id:0 for k in self.active_aide}

        for k in Solve:
            for d in range(self.instance.horizon):
                if len(f[k][d])>0:
                    Solve[k]=1
                    break
        for k in self.active_aide:

            if Solve[k.id]==1:
                logger.info("Final Solve Aide %s", k.id)
                logger.info("list Pik: %s", f[k.id])
                logger.info("list visit: %s", list_visit[k.id])
                logger.info("list Pik frequency: %s", list_Pik_frequency[k.id])
                subproblem=Subproblem(self.instance,k.id,f[k.id],list_visit[k.id],list_Pik_frequency[k.id],self.overtime,self.alpha,self.beta,self.TCov,self.mean_learning,self.std_learning)

                subproblem.generate_cut(True)

        self.frequency_visit={i:0 for i in range(1,self.instance.horizon+1)}
        self.total_visit={i:0 for i in range(1,self.instance.horizon+1)}

        for i in self.new_patient:
            if i.aide_assigned:

                for z in i.aide_assigned.values():
                    self.frequency_visit[z]+=1

                self.total_visit[i.number_visit]+=1

        self.write_solution(m.ObjVal,m._call,m._nblazy,m._subcall,m._complete_sub,m._inf_complete_sub,m._time_limit,m.Runtime,m._subTime,m.NodeCount)

    def branch_and_check(self,env):



        with Model(env=env) as m:

            #m=Model('Heching_master')
            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta") # whether a patient is acepted or not
            #x=m.addVars([(k, i) for i in self.list_patient_new_id for k in self.list_assignment[i]],vtype=GRB.BINARY, name="x")   # assignment of an aide to a patient over the horizon
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY, name="y") # daily assignment of an aide to a patient on requested days
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0, ub=self.horizon,name="num") # the number of times an aide k has served patient i at end of day d
            #pattern=m.addVars([(i,pattern_id) for i in self.list_patient_new_id for pattern_id in self.patient_pattern[i]],vtype=GRB.BINARY,name="pattern") # whether a patient is served according to a specific pattern or not
            freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq") # if aide k visited patient i on day d for the p th time

            # Fix delta=1 for returning patients from the previous run
            for i in self.past_aide_visits:
                if i in self.list_patient_new_id:
                    delta[i].lb=1

            # MIP warm start: complete hint for all integer/binary variables.
            # num is cumulative (sum of y up to day d for each aide), so we must set it
            # for ALL aides at every day — not just on the day they visit — otherwise
            # the number_visit constraint is violated for aides whose last visit was
            # on an earlier day.
            for k, i, d in self.day_assignment:
                num[k, i, d].Start = 0

            for i in self.list_patient_new_id:
                if not self.prev_day_assignment.get(i):
                    continue
                delta[i].Start = 1

                # Pass 1: set y hints for all (aide, day) pairs of this patient.
                for k, ii, d in self.day_assignment.select('*', i, '*'):
                    prev_k = self.prev_day_assignment[i].get(d)
                    y[k, ii, d].Start = 1 if k == prev_k else 0

                # Pass 2: set num and freq, accumulating running count per aide across days.
                running_count = {}  # {aide_k: cumulative visits through current day}
                for d in range(self.horizon):
                    if self.instance.list_patient[i].pre_assigned_day[d] != 1:
                        continue
                    prev_k = self.prev_day_assignment[i].get(d)
                    if prev_k is not None:
                        running_count[prev_k] = running_count.get(prev_k, 0) + 1
                    # num[k,i,d] = total visits by k to i through day d (all aides)
                    for k, ii, dd in self.day_assignment.select('*', i, d):
                        num[k, ii, dd].Start = running_count.get(k, 0)
                    # freq hint only for the visiting aide
                    if prev_k is not None and (prev_k, i, d) in self.day_assignment:
                        past_k = self.past_aide_visits.get(i, {}).get(prev_k, 0)
                        p_hint = past_k + running_count[prev_k]
                        if (prev_k, i, d, p_hint) in self.day_visit_frequency:
                            freq[prev_k, i, d, p_hint].Start = 1

            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id), GRB.MAXIMIZE)  #self.instance.list_patient[i].number_visit


            #m.addConstr(obj<=self.maximum_mean_service_time+self.maximum_travel_time,"maximum_objective")
            #m.addConstr(obj<=self.maximum_obj,"maximum_objective")

            #m.addConstrs((quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i])==delta[i] for i in self.list_patient_new_id),"pattern_selection")
            #m.addConstrs((quicksum(x[k,i] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id),"driverCon")


            #m.addConstrs((y[k,i,d]==quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i] if d in self.patient_pattern[i][pattern_id] and self.patient_pattern[i][pattern_id][d]==k) for k,i,d in self.day_assignment),"aide_day_pattern")
            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            #m.addConstrs((y[k,i,d]<=x[k,i] for k,i,d in self.day_assignment),"unique_assigned_aide")


            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")


            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationFrom[kk,i,p,dd] for i in self.morning_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=72-self.instance.list_aide[kk].time_window_start[0]
             #             for dd in range(self.horizon) for kk in self.list_aide_active_id),"morning_time")

            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationTo[kk,i,p,dd] for i in self.evening_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=self.instance.list_aide[kk].time_window_end[1]-72
                      #    for dd in range(self.horizon) for kk in self.list_aide_active_id),"evening_time")


            m.addConstrs((quicksum(freq[k,i,d,p]*(self.ugmentedDurationFrom[k,i,d,p]) for i in self.morning_patient for p in self.patient_day_frequency[i].get(d,[]) if (k,i,d,p) in self.day_visit_frequency)<=720-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")

            m.addConstrs((quicksum(freq[k,i,d,p]*(self.ugmentedDurationTo[k,i,d,p]) for i in self.evening_patient for p in self.patient_day_frequency[i].get(d,[]) if (k,i,d,p) in self.day_visit_frequency)<=self.instance.list_aide[k].time_window_end[1]-720
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")


            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            # patient_frequency_day is now per aide: {i: {k: {p: [days]}}}
            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][k][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i][k]),"frequency_unique")

            # (p - past_k) gives the local (new-run) visit index so we count only new visits
            m.addConstrs((quicksum((p-self.past_aide_visits.get(i,{}).get(k,0))*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            # comment lexio-graphy constraint to see the effect of solving time

            #m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][k][p] if dd<=d )<=
             #             quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][k][p-1] if dd<d )
              #            for i in self.list_patient_new_id for k in self.list_assignment[i]
               #           for p in self.patient_frequency_day[i][k] for d in self.patient_frequency_day[i][k][p]
                #          if p>self.past_aide_visits.get(i,{}).get(k,0)+1 and (p-1) in self.patient_frequency_day[i][k]),"order_frequency")

            #m.addConstrs((z[k,d]>=z[kk,d] for k,kk in self.dominance_pairs for d in range(self.horizon)),"dominance")

            #m.addConstrs((quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d1))<=quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d2)) for d1 in range(self.horizon) for d2 in range(d1+1,self.horizon) for kk in range(self.nb_aide)),"continuity")
            
            #m.addConstrs((y[k.id,i.id,d1]<=y[k.id,i.id,d2] for i in self.new_patient for k in self.active_aide for d1 in range(self.horizon) 
             #             for d2 in range(d1+1,self.horizon) if ((k.id,i.id,d1) in self.day_assignment) and ((k.id,i.id,d2) in self.day_assignment)),"continuity")

            
            #extra time_window constraints >>>it makes the solving time worse 

            """
            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_aide[k].time_window_start[0]+self.ugmentedDurationFrom[k,i,d]>self.instance.list_patient[i].time_window[1]),"time_window_morning")


            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_patient[i].time_window[0]+self.ugmentedDurationTo[k,i,d]>self.instance.list_aide[k].time_window_end[1]),"time_window_evening")

            m.addConstrs((quicksum(x[k,i]*(self.instance.list_patient[i].visit_duration)*(self.instance.list_patient[i].number_visit)
                                   for k,i in self.all_assignment.select(kk,'*'))<=self.instance.list_aide[kk].maximum_work_time for kk in range(self.nb_aide)),"over_time")

            """
            #m.write("heching-master.lp")self.instance.list_aide[k].time_window_end[0]


            m._yvars = y
            m._freqvars = freq
            m._numvars = num
            m._instance=self.instance
            m._activeAides=self.active_aide
            m._assignment=self.day_assignment
            m._freqAssignment=self.day_visit_frequency
            m._call=0
            m._nblazy=0
            m._subcall=0
            m._subTime=0
            m._complete_sub=0
            m._inf_complete_sub=0
            m._time_limit=0
            #m._objvar=obj
            #m._scenario=create_scenario(self.instance,self.nb_scenario)
            m._overtime=self.overtime
            m._alpha=self.alpha
            m._beta=self.beta
            m._TCov=self.TCov
            m._meanLearning=self.mean_learning
            m._stdLearning=self.std_learning
            m._similarFirst=self.similar_first
            m.setParam("LazyConstraints",1)
            m.setParam("TimeLimit",3600)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            #m.setParam("MIPFocus",1)
            m.setParam("Heuristics",0)
            m.setParam("Cuts",0)
            #m.setParam("LogFile","log_file_master_approx.txt")
            #print(m.getParamInfo("TimeLimit"))
            #print(m.getAttr("NumVars"), m.getAttr("NumConstrs"))
            
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path=os.path.join(self.BASE_DIR, 'result','learn',  f"{self.file_name}-Learning-TimeWindowRelaxation-EVT.lp")
            log_path=os.path.join(self.BASE_DIR, 'result','learn',  f"{self.file_name}-Learning-TimeWindowRelaxation-EVT.log")
            m.setParam("LogFile", log_path)
            m.write(file_path)

            m.optimize(lazy_callback)
            if m.status == GRB.OPTIMAL:

                logger.info("--------------------------------")
                logger.info("Optimal solution found with objective %s", m.ObjVal)
                logger.info("--------------------------------")

                self._finalize_solution(m,y,num,freq)
                return m.Runtime,m.ObjVal
            elif m.status == GRB.TIME_LIMIT:
                #print("Time limit reached for master problem")
                logger.warning("Time limit %s reached for master problem", m.Runtime)
                if m.SolCount>0:
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    logger.info("the best bound is: %s", m.ObjBound)
                    logger.info("the gap is: %s", m.MIPGap)
                    self._finalize_solution(m,y,num,freq)
                    return m.Runtime,m.ObjVal
                else:
                    #print("No feasible solution found")
                    logger.warning("No feasible solution found")
                    return m.Runtime, 0
            elif m.status ==GRB.INFEASIBLE:
                #print("The problem is infeasible")
                logger.error("The problem is infeasible")

                m.computeIIS()
                file_path=os.path.join(self.BASE_DIR, 'result','learn', 'infeasible.ilp')
                m.write(file_path)
                return 0, 0
            elif m.status== GRB.INTERRUPTED:
                #print("The optimization is interrupted")
                logger.error("The optimization is interrupted")
                if m.SolCount>0:
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    logger.info("the best bound is: %s", m.ObjBound)
                    logger.info("the gap is: %s", m.MIPGap)
                    self._finalize_solution(m,y,num,freq)
                    return m.Runtime,m.ObjVal
                else:
                    #print("No feasible solution found")
                    logger.warning("No feasible solution found")
                    return m.Runtime, 0
                
                        
                                    
                        
        
        
        
    
    def logic_based_benders(self,env):

        TIME_LIMIT=14400
        start_time=time.time()

        with Model(env=env) as m:

            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta")
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY,name="y")
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0,ub=self.horizon,name="num")
            freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq")

            # Fix delta=1 for returning patients from the previous run
            for i in self.past_aide_visits:
                if i in self.list_patient_new_id:
                    delta[i].lb=1

            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id),GRB.MAXIMIZE)

            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")

            m.addConstrs((quicksum(freq[k,i,d,p]*(self.ugmentedDurationFrom[k,i,d,p]) for i in self.morning_patient for p in self.patient_day_frequency[i].get(d,[]) if (k,i,d,p) in self.day_visit_frequency)<=720-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")

            m.addConstrs((quicksum(freq[k,i,d,p]*(self.ugmentedDurationTo[k,i,d,p]) for i in self.evening_patient for p in self.patient_day_frequency[i].get(d,[]) if (k,i,d,p) in self.day_visit_frequency)<=self.instance.list_aide[k].time_window_end[1]-720
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")

            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            # patient_frequency_day is now per aide: {i: {k: {p: [days]}}}
            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][k][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i][k]),"frequency_unique")

            # (p - past_k) gives the local (new-run) visit index so we count only new visits
            m.addConstrs((quicksum((p-self.past_aide_visits.get(i,{}).get(k,0))*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            # order_frequency: enforce visit ordering only among new-run visits (skip when p-1 is a past visit)
            m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][k][p] if dd<=d)<=
                          quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][k][p-1] if dd<d)
                          for i in self.list_patient_new_id for k in self.list_assignment[i]
                          for p in self.patient_frequency_day[i][k] for d in self.patient_frequency_day[i][k][p]
                          if p>self.past_aide_visits.get(i,{}).get(k,0)+1 and (p-1) in self.patient_frequency_day[i][k]),"order_frequency")

            m.setParam("TimeLimit",TIME_LIMIT)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            m.setParam("MIPFocus",1)
            m.setParam("Heuristics",0.3)

            self.BASE_DIR=os.path.dirname(os.path.abspath(__file__))
            lp_path=os.path.join(self.BASE_DIR,'result','learn',f"{self.file_name}-Learning-LBBD-EVT.lp")
            log_path=os.path.join(self.BASE_DIR,'result','learn',f"{self.file_name}-Learning-LBBD-EVT.log")
            m.setParam("LogFile",log_path)
            m.write(lp_path)

            nb_iter=0
            nb_cuts=0
            nb_subcall=0
            sub_time_total=0.0
            complete_sub_total=0
            inf_sub_total=0

            while True:
                elapsed=time.time()-start_time
                remaining=TIME_LIMIT-elapsed
                if remaining<=0:
                    logger.warning("LBBD total time limit reached before re-solve")
                    return time.time()-start_time,0
                m.setParam("TimeLimit",remaining)

                m.optimize()
                nb_iter+=1

                if m.status==GRB.INFEASIBLE:
                    logger.error("LBBD: master infeasible")
                    m.computeIIS()
                    m.write(os.path.join(self.BASE_DIR,'result','learn','infeasible-lbbd.ilp'))
                    return 0,0

                if m.SolCount==0:
                    logger.warning("LBBD: no solution at iteration %s",nb_iter)
                    return time.time()-start_time,0

                if m.status==GRB.TIME_LIMIT:
                    logger.warning("LBBD: time limit at iteration %s",nb_iter)
                    if m.SolCount>0:
                        logger.info("best obj %s bound %s gap %s",m.ObjVal,m.ObjBound,m.MIPGap)
                    return time.time()-start_time,0

                if m.status==GRB.INTERRUPTED:
                    logger.warning("LBBD: interrupted at iteration %s",nb_iter)
                    if m.SolCount>0:
                        logger.info("best obj %s",m.ObjVal)
                    return time.time()-start_time,0

                logger.info("LBBD iteration %s obj=%s",nb_iter,m.ObjVal)

                list_Pik={k.id:{d:[i for kk,i,dd in self.day_assignment.select(k.id,'*','*') if dd==d and y[kk,i,dd].X>0.5] for d in
                                 range(self.instance.horizon)} for k in self.active_aide}
                list_visit={k.id:{d:{i:int(np.round(num[kk,i,dd].X)) for kk,i,dd in self.day_assignment.select(k.id,'*','*') if dd==d and y[kk,i,dd].X>0.5} for d in
                                 range(self.instance.horizon)} for k in self.active_aide}
                list_Pik_frequency={k.id:{d:{i:p for kk,i,dd,p in self.day_visit_frequency.select(k.id,'*','*','*') if dd==d and freq[kk,i,dd,p].X>0.5} for d in
                                 range(self.instance.horizon)} for k in self.active_aide}

                all_feasible=True
                for k in self.active_aide:
                    if not any(len(list_Pik[k.id][d])>0 for d in range(self.instance.horizon)):
                        continue

                    value=""
                    for d in range(self.instance.horizon):
                        if list_Pik[k.id][d]:
                            for b in list_Pik[k.id][d]:
                                value+=str(b)+"({})".format(list_Pik_frequency[k.id][d][b])+'-'
                            value+="/"

                    if value in self.instance.list_feasible_per_aide[k.id]:
                        continue

                    logger.info("LBBD checking aide %s",k.id)
                    subproblem=Subproblem(self.instance,k.id,list_Pik[k.id],list_visit[k.id],list_Pik_frequency[k.id],
                                         self.overtime,self.alpha,self.beta,self.TCov,self.mean_learning,self.std_learning)
                    nb_subcall+=1
                    t0=time.time()
                    over_time_status,status,solve_daily_sub,inf_daily_sub,complete_sub,dic_cut=subproblem.generate_cut(False)
                    sub_time_total+=time.time()-t0
                    complete_sub_total+=complete_sub

                    if status==False:
                        inf_sub_total+=1
                        all_feasible=False
                        for d in dic_cut:
                            if len(dic_cut[d])>0:
                                m.addConstr(quicksum((1-freq[k.id,i,d,p]) for (i,p) in dic_cut[d])>=1)
                                nb_cuts+=1
                                logger.info("LBBD cut aide %s day %s: %s",k.id,d,dic_cut[d])

                if all_feasible:
                    logger.info("LBBD converged: iter=%s cuts=%s obj=%s",nb_iter,nb_cuts,m.ObjVal)
                    best_obj=m.ObjVal
                    m.write(os.path.join(self.BASE_DIR,'result','learn',f"{self.file_name}-Learning-LBBD-solution-EVT.sol"))

                    f_sol={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].X>0.5] for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}
                    list_visit_sol={k:{d:{i:int(np.round(num[kk,i,dd].X)) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].X>0.5} for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}
                    list_Pik_freq_sol={k:{d:{i:p for kk,i,dd,p in self.day_visit_frequency.select(k,'*','*','*') if dd==d and freq[kk,i,dd,p].X>0.5} for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}

                    Solve={k.id:0 for k in self.active_aide}
                    for k_id in Solve:
                        for d in range(self.instance.horizon):
                            if len(f_sol[k_id][d])>0:
                                Solve[k_id]=1
                                break
                    for k in self.active_aide:
                        if Solve[k.id]==1:
                            logger.info("LBBD final solve aide %s",k.id)
                            subproblem=Subproblem(self.instance,k.id,f_sol[k.id],list_visit_sol[k.id],list_Pik_freq_sol[k.id],
                                                  self.overtime,self.alpha,self.beta,self.TCov,self.mean_learning,self.std_learning)
                            subproblem.generate_cut(True)

                    self.frequency_visit={i:0 for i in range(1,self.instance.horizon+1)}
                    self.total_visit={i:0 for i in range(1,self.instance.horizon+1)}
                    for i in self.new_patient:
                        if i.aide_assigned:
                            for z in i.aide_assigned.values():
                                self.frequency_visit[z]+=1
                            self.total_visit[i.number_visit]+=1

                    total_time=time.time()-start_time
                    self.write_lbbd_solution(best_obj,nb_iter,nb_cuts,nb_subcall,complete_sub_total,inf_sub_total,0,total_time,sub_time_total,m.NodeCount)
                    return total_time,best_obj

    def write_lbbd_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','learn')
        with open(os.path.join(file_path,f"{self.file_name}-Learning-LBBD-output_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and LBBD \n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("TCov {}\n".format(self.TCov))
            f.write("MeanLearningDuration {}\n".format(self.mean_learning))
            f.write("StdLearningDuration {}\n".format(self.std_learning))
            f.write("PatientServiceLevel {}\n".format(self.alpha))
            f.write("AideServiceLevel {}\n".format(self.beta))
            f.write("Gamma1 {}\n".format(self.gamma1))
            f.write("Gamma2 {}\n".format(self.gamma2))
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("LBBDIteration {}".format(nbcall))
            f.write('\n')
            f.write("FeasibilityCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))

        with open(os.path.join(file_path,f"{self.file_name}-Learning-LBBD-route_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and LBBD \n")
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-Learning-LBBD-schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and LBBD \n")
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-Learning-LBBD-Route-Schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and LBBD \n")
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-Learning-LBBD-Patient-CoC_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and LBBD \n")
            for i in self.new_patient:
                if i.aide_assigned:
                    f.write('Patient {}'.format(i.id))
                    f.write('\n')
                    f.write('Aide(number of visits)')
                    f.write('\n')
                    f.write(str(i.aide_assigned))
                    f.write('\n')
                    f.write('\n')
            for i,j in self.frequency_visit.items():
                f.write('Actual number of visits {}: {} patients'.format(i,self.total_visit[i]))
                f.write('\n')
                f.write('Assigned Number of visits {}: {} patients'.format(i,j))
                f.write('\n')

    def write_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','learn')
        #"""
        with open(os.path.join(file_path,f"{self.file_name}-Learning-TimeWindowRelaxation-output_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and TimeWindowRelaxation \n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("TCov {}\n".format(self.TCov))
            f.write("MeanLearningDuration {}\n".format(self.mean_learning))
            f.write("StdLearningDuration {}\n".format(self.std_learning))
            
            f.write("PatientServiceLevel {}\n".format(self.alpha))
            f.write("AideServiceLevel {}\n".format(self.beta))
            f.write("Gamma1 {}\n".format(self.gamma1))
            f.write("Gamma2 {}\n".format(self.gamma2))
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("CallbackCall {}".format(nbcall))
            f.write('\n')
            f.write("LazyCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            #f.write("InfeasibleSubproblemsReachingTimeLimit {} \n".format(time_limit))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))
            
            
            #f.write("patient_id,aide_id,visit_time,assigned_aide_day_1,assigned_aide_day_2,assigned_aide_day_3,assigned_aide_day_4,assigned_aide_day_5")
            #f.write('\n')
            #for i in self.list_patient_new_id:
             #   f.write(str(self.instance.list_patient[i].id)+','+str(self.instance.list_patient[i].pre_assigned_aide[0])+
              #          ','+str(self.instance.list_patient[i].pre_assigned_aide[1])+','+
               #        str(self.instance.list_patient[i].pre_assigned_aide[2])+','+str(self.instance.list_patient[i].pre_assigned_aide[3])+','+
                #       str(self.instance.list_patient[i].pre_assigned_aide[4]))
                #f.write('\n')
                
        #"""


        with open(os.path.join(file_path,f"{self.file_name}-Learning-TimeWindowRelaxation-route_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and TimeWindowRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-Learning-TimeWindowRelaxation-schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and TimeWindowRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                


        with open(os.path.join(file_path,f"{self.file_name}-Learning-TimeWindowRelaxation-Route-Schedule_EVT.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and TimeWindowRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')
        

        with open(os.path.join(file_path,f"{self.file_name}-Learning-TimeWindowRelaxation-Patient-CoC_EVT.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")

            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and TimeWindowRelaxation \n")

            for i in self.new_patient:
                if i.aide_assigned:
                    f.write('Patient {}'.format(i.id))
                    f.write('\n')
                    f.write('Aide(number of visits)')
                    f.write('\n')
                    f.write(str(i.aide_assigned))
                    f.write('\n')
                    f.write('\n')
            
            for i,j in self.frequency_visit.items():
                f.write('Actual number of visits {}: {} patients'.format(i,self.total_visit[i]))
                f.write('\n')
                f.write('Assigned Number of visits {}: {} patients'.format(i,j))
                f.write('\n')


        
        
                    
                    

            
        
        #"""
    

def no_learn_lazy_callback(model,where):
    
    if where==GRB.Callback.MIPSOL:

        #print("Current time: {}".format(grb_t))
        """
        if grb_t >= 240:
            model.terminate()
            return

        """

        model._call+=1
           
        yval=model.cbGetSolution(model._yvars)
        numval=model.cbGetSolution(model._numvars)
        
        
        list_Pik={k.id:{d:[i for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5] for d in 
                                 range(model._instance.horizon)} for k in model._activeAides}
        
        
        list_visit={k.id:{d:{i:int(np.round(numval[kk,i,dd])) for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5} for d in 
                                 range(model._instance.horizon)} for k in model._activeAides}
        

        
        #can_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        

        #print("the candidate objective is {}".format(model.cbGetSolution(model._objvar)))

    
        #print("the list of Pik is:")
        #print(list_Pik)
        
        Solve={k.id:0 for k in model._activeAides}
        
        for k in Solve:
            for d in range(model._instance.horizon):
                if len(list_Pik[k][d])>0:
                    Solve[k]=1
                    break
        for k in model._activeAides:
            
            if Solve[k.id]==1:
                #print("----------------------------------")
                logger.info("----------------------------------")
                #print("Solve Aide {}".format(k.id))
                logger.info("Solve Aide %s ", k.id)

                #print("the list of Pik is:")
                logger.info("list Pik %s", list_Pik[k.id])
                logger.info("list visit %s", list_visit[k.id])
                #print(list_Pik[k.id])

                #print("the list of visit is:")

                #print(list_visit[k.id])

                """
                assignment_pattern=[]    
                for d in range(model._instance.horizon):
                    if list_Pik[k.id][d] not in assignment_pattern and list_Pik[k.id][d]:
                        assignment_pattern.append(list_Pik[k.id][d])
                        
                number_pattern=len(assignment_pattern)

                """
                value=""
                for d in range(model._instance.horizon):
                    if list_Pik[k.id][d]:
                        for b in list_Pik[k.id][d]:
                            value+=str(b)+'-'
                        value+="/"
                    
                    
                
                if value not in model._instance.list_feasible_per_aide[k.id]:
                    
                    #print("the assigned patients to aide {}".format(k.id+1))
                    #print(list_Pik[k.id])
                    subproblem=No_Learn_Subproblem(model._instance,k.id,list_Pik[k.id],list_visit[k.id],model._overtime, model._alpha,model._beta,model._TCov,model._meanNoLearn,model._stdNoLearn)
                    model._subcall+=1
                    start=time.time()
                    over_time_status,status,solve_daily_sub,inf_daily_sub,complete_sub,dic_cut=subproblem.generate_cut(False)
                    end=time.time()
                    total_time=end-start
                    model._subTime+=total_time
                    model._complete_sub+=complete_sub
                    
                    if status==False:
                        
                        model._inf_complete_sub+=1   # increase the number of sub-problems solve completely and are infeasible
                        """
                        for d in range(model._instance.horizon):
                            if len(list_cut[d])>0:
                                model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i in list_cut[d])>=1)
                                model._nblazy+=1 
                        """
                        for d in dic_cut:
                            if len(dic_cut[d])>0:
                                model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i in dic_cut[d])>=1)
                                model._nblazy+=1
                        
                        #model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i,d in list_cut)>=1)

                        
                        #print("the list of cut is:")
                        logger.info("Feasibility cut added %s", dic_cut)
                        #print(dic_cut)
                        #print("----------------------------------")
                        logger.info("----------------------------------")



class No_Learn_Master_Problem(): # without learning effect and use only time window relaxation

    def __init__(self,file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov,team,team_size,mean_no_learn,std_no_learn,returning_patients=None):

        #print("Master Problem is created")

        self.instance=instance
        self.overtime=overtime
        self.alpha=alpha
        self.beta=beta
        self.gamma1=gamma1
        self.gamma2=gamma2
        self.TCov=TCov
        self.team_size=team_size
        self.file_name=file_name
        # returning_patients: set of patient IDs accepted in the previous run.
        # delta is fixed to 1 for these patients — no visit history needed for no-learn.
        self.returning_patients = returning_patients if returning_patients is not None else set()
        self.meanNoLearn=mean_no_learn
        self.stdNoLearn=std_no_learn

        
        
        
        
        #new patient are those who are available to be assigned to an aide
        self.new_patient=[instance.list_patient[i] for i in range(instance.number_patient) if instance.list_patient[i].
                            availability==2]
        
        
        #active aide are those who are available to serve patients
        self.active_aide=[instance.list_aide[i] for i in range(instance.number_aide) if instance.list_aide[i].
                            availability==2]
        
        self.list_aide_active_id=[]
        for k in self.active_aide:
            self.list_aide_active_id.append(k.id)

        self.list_patient_new_id=[]
        for i in self.new_patient:
            self.list_patient_new_id.append(i.id)

        #self.nb_patient=len(self.new_patient) 
        #self.nb_aide=len(self.active_aide)  
        

        self.horizon=instance.horizon
        
        if team:
            self.team_maker()

            
            #print("team maker is done")
            logger.info("Team maker is done")

            #import pdb; pdb.set_trace()

            self.day_assignment=tuplelist([(k,i,d) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1])

            self.day_visit_frequency=tuplelist([(k,i,d,p) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i] for p in self.patient_visit_frequency[i][d][k]])


        else:
            self.day_assignment=tuplelist([(k.id,i.id,d) for k in self.active_aide for i in self.new_patient for d in range(self.horizon) if i.pre_assigned_day[d]==1])
        
        #to calculate maximum number of potential visits for each (k,i,d)
        
        
        
        #self.new_assignment=tuplelist([(k.id,i.id) for k in self.active_aide for i in self.free_patient])
        #print("the potential patient-aide assignment") 
        #print(self.new_assignment)
        #self.all_assignment=self.current_assignment+self.new_assignment
        #print("all assignments:")
        #print(self.all_assignment)
        
        self.morning_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[1]<=720]
        self.evening_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[0]>=720]

        #print("morning patient are: {}".format(self.morning_patient))
        logger.info("morning patient are: %s", self.morning_patient)
        #print("evening patient are: {}".format(self.evening_patient))
        logger.info("evening patient are: %s", self.evening_patient)

        #self.ugmentedDurationFrom={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_from[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}
        
        #self.ugmentedDurationTo={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_to[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}



        self.ugmentedDurationFrom={(k,i,d):(self.meanNoLearn[0]+self.gamma1*self.stdNoLearn[0])+self.min_travel_from[i][d][k] for k,i,d in self.day_assignment}

        self.ugmentedDurationTo={(k,i,d):(self.meanNoLearn[0]+self.gamma2*self.stdNoLearn[0])+self.min_travel_to[i][d][k] for k,i,d in self.day_assignment}

                   
                            
    
    """ It is a function to call the subproblem 
        when the master problem finds a feasible solution 
        and add feasibility cuts when they are required.
        It links the master and subproblem. 
    """
    
    def team_maker(self):
        random.seed(42)  # Set the seed once for reproducibility

        self.list_assignment={}
        for i in self.list_patient_new_id:

            self.list_assignment[i]=[]

            potential_aide=[]

            earliest_start=self.instance.list_patient[i].time_window[0]
            latest_start=self.instance.list_patient[i].time_window[1]-self.instance.list_patient[i].visit_duration
            visit_duration=self.instance.list_patient[i].visit_duration


            for k in self.list_aide_active_id:
                travel_time=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                time_window_start=self.instance.list_aide[k].time_window_start[0]
                time_window_end=self.instance.list_aide[k].time_window_end[1]

                if time_window_start + travel_time > latest_start:  # cannot reach the patient before his/her latest start time
                    continue
                elif travel_time+visit_duration > time_window_end + self.overtime - earliest_start:  
                    continue
                elif earliest_start + visit_duration + self.beta * (0.25 * visit_duration) > time_window_end + self.overtime:  
                    continue
                else:
                    potential_aide.append(k)

            if len(potential_aide)<=self.team_size:
                self.list_assignment[i]=potential_aide
            else:
                selected_aide=random.sample(potential_aide, self.team_size)
                self.list_assignment[i]=selected_aide

        self.patient_pattern={}

        for i in self.list_patient_new_id:
            self.patient_pattern[i]={}
            visit_days=[ d for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1]
            patterns=list(product(self.list_assignment[i], repeat=len(visit_days)))
            self.patient_pattern[i]={n:{day:p[d] for d,day in enumerate(visit_days)} for n,p in enumerate(patterns)}

        self.aide_day_patient={}
        for k in self.list_aide_active_id:
            self.aide_day_patient[k]={}
            for d in range(self.horizon):
                self.aide_day_patient[k][d]=[]
                for i in self.list_patient_new_id:
                    if self.instance.list_patient[i].pre_assigned_day[d]==1 and k in self.list_assignment[i]:
                        self.aide_day_patient[k][d].append(i)
        self.min_travel_from={}
        self.min_travel_to={}

        for i in self.list_patient_new_id:
            self.min_travel_from[i]={}
            self.min_travel_to[i]={}
            for d in range(self.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    self.min_travel_from[i][d]={}
                    self.min_travel_to[i][d]={}
                    for k in self.list_assignment[i]:
                        minTravelFrom=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]<minTravelFrom:
                                minTravelFrom=self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]
                        self.min_travel_from[i][d][k]=minTravelFrom

                        minTravelTo=100
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]<minTravelTo:
                                minTravelTo=self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]
                        if minTravelTo==100:
                            minTravelTo=0
                        self.min_travel_to[i][d][k]=minTravelTo
    
        
        self.patient_pattern_duration={}
        for i in self.list_patient_new_id:
            self.patient_pattern_duration[i]={}
            visit_duration=self.instance.list_patient[i].visit_duration
            for pattern_id,pattern in self.patient_pattern[i].items():
                self.patient_pattern_duration[i][pattern_id]={}
                aide_assignment_count={}
                for day,aide in pattern.items():
                    if aide not in aide_assignment_count:
                        aide_assignment_count[aide]=0
                    
                    aide_assignment_count[aide]+=1
                    learning_index=aide_assignment_count[aide]-1

                    self.patient_pattern_duration[i][pattern_id][day]=int(np.round(visit_duration))

        
        self.patient_visit_frequency={}
        self.patient_duration_frequency={}
        self.patient_day_frequency={}
        for i in self.list_patient_new_id:
            visit_duration=self.instance.list_patient[i].visit_duration
            self.patient_visit_frequency[i]={}
            self.patient_duration_frequency[i]={}
            self.patient_day_frequency[i]={}
            total_visit=0
            for d in range(self.instance.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    total_visit+=1
                    self.patient_duration_frequency[i][total_visit]=int(np.round((visit_duration)+1.64*(visit_duration*0.25)))
                    self.patient_visit_frequency[i][d]={}
                    self.patient_day_frequency[i][d]=[]
                    for k in self.list_assignment[i]:
                        self.patient_visit_frequency[i][d][k]=[]
                        for p in range(total_visit):
                            self.patient_visit_frequency[i][d][k].append(p+1)
                    for p in range(total_visit):
                        self.patient_day_frequency[i][d].append(p+1)
    
        
        self.patient_frequency_day={}

        for i in self.list_patient_new_id:
            self.patient_frequency_day[i]={}
            for p in self.patient_duration_frequency[i]:
                self.patient_frequency_day[i][p]=[]
                index=0
                for d in range(self.instance.horizon):
                    if self.instance.list_patient[i].pre_assigned_day[d]==1:
                        index+=1
                        if index>=p:
                            self.patient_frequency_day[i][p].append(d)
        
        """
        print("patient frequency day:")
        for i in self.list_patient_new_id:
            for p in self.patient_frequency_day[i]:
                print("patient {}, frequency {}, days {}".format(i,p,self.patient_frequency_day[i][p]))

        """


                    
        
        
    
    
    def _finalize_solution(self,m,y,num):
        """Extract the current incumbent from m and run the final exact
        subproblem solve (generate_cut(True)) for every aide with a visit,
        to obtain schedules/routes. Valid for any status as long as
        m.SolCount>0 (OPTIMAL, or TIME_LIMIT/INTERRUPTED with a feasible
        incumbent), since .X reads the incumbent regardless of status.
        """

        file_path=os.path.join(self.BASE_DIR, 'result','no-learn', f"{self.file_name}-NoLearning-TimeWindowRelaxation-solution-EVT.sol")
        m.write(file_path)
        f={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5] for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}

        list_visit={k:{d:{i:int(np.round(num[kk,i,dd].getAttr('X'))) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5} for d in
                                 range(self.horizon)} for k in self.list_aide_active_id}

        Solve={k.id:0 for k in self.active_aide}

        for k in Solve:
            for d in range(self.instance.horizon):
                if len(f[k][d])>0:
                    Solve[k]=1
                    break
        for k in self.active_aide:

            if Solve[k.id]==1:
                logger.info("Final Solve Aide %s", k.id)
                logger.info("list Pik %s", f[k.id])

                subproblem=No_Learn_Subproblem(self.instance,k.id,f[k.id],list_visit[k.id],self.overtime,self.alpha,self.beta,self.TCov,self.meanNoLearn,self.stdNoLearn)

                subproblem.generate_cut(True)

        self.frequency_visit={i:0 for i in range(1,self.instance.horizon+1)}
        self.total_visit={i:0 for i in range(1,self.instance.horizon+1)}

        for i in self.new_patient:
            if i.aide_assigned:

                for z in i.aide_assigned.values():
                    self.frequency_visit[z]+=1

                self.total_visit[i.number_visit]+=1

        self.write_solution(m.ObjVal,m._call,m._nblazy,m._subcall,m._complete_sub,m._inf_complete_sub,m._time_limit,m.Runtime,m._subTime,m.NodeCount)

    def branch_and_check(self,env):



        with Model(env=env) as m:

            #m=Model('Heching_master')
            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta") # whether a patient is acepted or not
            #x=m.addVars([(k, i) for i in self.list_patient_new_id for k in self.list_assignment[i]],vtype=GRB.BINARY, name="x")   # assignment of an aide to a patient over the horizon
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY, name="y") # daily assignment of an aide to a patient on requested days
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0, ub=self.horizon,name="num") # the number of times an aide k has served patient i at end of day d
            #pattern=m.addVars([(i,pattern_id) for i in self.list_patient_new_id for pattern_id in self.patient_pattern[i]],vtype=GRB.BINARY,name="pattern") # whether a patient is served according to a specific pattern or not
            #freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq") # if aide k visited patient i on day d for the p th time


            # Fix delta=1 for patients accepted in the previous run (no history needed for no-learn)
            for i in self.returning_patients:
                if i in self.list_patient_new_id:
                    delta[i].lb=1
            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id), GRB.MAXIMIZE)  #self.instance.list_patient[i].number_visit


            
            #m.addConstr(obj<=self.maximum_mean_service_time+self.maximum_travel_time,"maximum_objective")
            #m.addConstr(obj<=self.maximum_obj,"maximum_objective")

            #m.addConstrs((quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i])==delta[i] for i in self.list_patient_new_id),"pattern_selection")
            #m.addConstrs((quicksum(x[k,i] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id),"driverCon")


            #m.addConstrs((y[k,i,d]==quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i] if d in self.patient_pattern[i][pattern_id] and self.patient_pattern[i][pattern_id][d]==k) for k,i,d in self.day_assignment),"aide_day_pattern")
            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            #m.addConstrs((y[k,i,d]<=x[k,i] for k,i,d in self.day_assignment),"unique_assigned_aide")


            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")


            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationFrom[kk,i,p,dd] for i in self.morning_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=72-self.instance.list_aide[kk].time_window_start[0] 
             #             for dd in range(self.horizon) for kk in self.list_aide_active_id),"morning_time")

            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationTo[kk,i,p,dd] for i in self.evening_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=self.instance.list_aide[kk].time_window_end[1]-72 
                      #    for dd in range(self.horizon) for kk in self.list_aide_active_id),"evening_time")


            m.addConstrs((quicksum(y[k,i,d]*(self.ugmentedDurationFrom[k,i,d]) for i in self.morning_patient if (k,i,d) in self.day_assignment)<=720-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")

            m.addConstrs((quicksum(y[k,i,d]*(self.ugmentedDurationTo[k,i,d]) for i in self.evening_patient if (k,i,d) in self.day_assignment)<=self.instance.list_aide[k].time_window_end[1]-720
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")

            """ Comment all constraints related to frequency. No need in no-learning case. 
            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i]),"frequency_unique")
            
            m.addConstrs((quicksum((p)*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][p] if dd<=d )<= 
                          quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][p-1] if dd<d ) 
                          for i in self.list_patient_new_id for k in self.list_assignment[i] 
                          for p in self.patient_frequency_day[i] for d in self.patient_frequency_day[i][p] if p>1 ),"order_frequency")
            """
            #m.addConstrs((z[k,d]>=z[kk,d] for k,kk in self.dominance_pairs for d in range(self.horizon)),"dominance")

            #m.addConstrs((quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d1))<=quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d2)) for d1 in range(self.horizon) for d2 in range(d1+1,self.horizon) for kk in range(self.nb_aide)),"continuity")
            
            #m.addConstrs((y[k.id,i.id,d1]<=y[k.id,i.id,d2] for i in self.new_patient for k in self.active_aide for d1 in range(self.horizon) 
             #             for d2 in range(d1+1,self.horizon) if ((k.id,i.id,d1) in self.day_assignment) and ((k.id,i.id,d2) in self.day_assignment)),"continuity")

            
            #extra time_window constraints >>>it makes the solving time worse 

            """
            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_aide[k].time_window_start[0]+self.ugmentedDurationFrom[k,i,d]>self.instance.list_patient[i].time_window[1]),"time_window_morning")


            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_patient[i].time_window[0]+self.ugmentedDurationTo[k,i,d]>self.instance.list_aide[k].time_window_end[1]),"time_window_evening")

            m.addConstrs((quicksum(x[k,i]*(self.instance.list_patient[i].visit_duration)*(self.instance.list_patient[i].number_visit)
                                   for k,i in self.all_assignment.select(kk,'*'))<=self.instance.list_aide[kk].maximum_work_time for kk in range(self.nb_aide)),"over_time")

            """
            #m.write("heching-master.lp")self.instance.list_aide[k].time_window_end[0]


            m._yvars = y
            #m._freqvars = freq
            m._numvars = num
            m._instance=self.instance
            m._activeAides=self.active_aide
            m._assignment=self.day_assignment
            m._freqAssignment=self.day_visit_frequency
            m._call=0
            m._nblazy=0
            m._subcall=0
            m._subTime=0
            m._complete_sub=0
            m._inf_complete_sub=0
            m._time_limit=0
            #m._objvar=obj
            #m._scenario=create_scenario(self.instance,self.nb_scenario)
            m._overtime=self.overtime
            m._alpha=self.alpha
            m._beta=self.beta
            m._TCov=self.TCov
            m._meanNoLearn=self.meanNoLearn
            m._stdNoLearn=self.stdNoLearn
            m.setParam("LazyConstraints",1)
            m.setParam("TimeLimit",3600)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            #m.setParam("MIPFocus",1)
            m.setParam("Heuristics",0)
            m.setParam("Cuts",0)
            #m.setParam("LogFile","log_file_master_approx.txt")
            #print(m.getParamInfo("TimeLimit"))
            #print(m.getAttr("NumVars"), m.getAttr("NumConstrs"))
            
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path=os.path.join(self.BASE_DIR, 'result','no-learn',  f"{self.file_name}-NoLearning-TimeWindowRelaxation-EVT.lp")
            log_path=os.path.join(self.BASE_DIR, 'result','no-learn', f"{self.file_name}-NoLearning-TimeWindowRelaxation-EVT.log")
            m.setParam("LogFile", log_path)
            m.write(file_path)
            
            m.optimize(no_learn_lazy_callback)
            if m.status == GRB.OPTIMAL:

                logger.info("--------------------------------")
                logger.info("Optimal solution found with objective %s", m.ObjVal)
                logger.info("--------------------------------")

                self._finalize_solution(m,y,num)
                return m.Runtime,m.ObjVal
            elif m.status == GRB.TIME_LIMIT:
                #print("Time limit reached for master problem")
                logger.error("Time limit %s reached for master problem", m.Runtime)
                if m.SolCount>0:
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    logger.info("the best bound is: %s", m.ObjBound)
                    logger.info("the gap is: %s", m.MIPGap)
                    self._finalize_solution(m,y,num)
                    return m.Runtime,m.ObjVal
                else:
                    #print("No feasible solution found")
                    logger.warning("No feasible solution found")
                    return m.Runtime, 0
            elif m.status ==GRB.INFEASIBLE:
                #print("The problem is infeasible")
                logger.error("The problem is infeasible")

                m.computeIIS()
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn',  'infeasible.ilp')
                m.write(file_path)
                return 0, 0
            elif m.status== GRB.INTERRUPTED:
                #print("The optimization is interrupted")
                logger.error("The optimization is interrupted")
                if m.SolCount>0:
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    logger.info("the best bound is: %s", m.ObjBound)
                    logger.info("the gap is: %s", m.MIPGap)
                    self._finalize_solution(m,y,num)
                    return m.Runtime,m.ObjVal
                else:
                    logger.warning("No feasible solution found")
                    #print("No feasible solution found")
                    return m.Runtime, 0
                
    def write_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','no-learn')
        #"""
        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-TimeWindowRelaxation-output_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("TCov {}\n".format(self.TCov))
            f.write("MeanNoLearn {}\n".format(self.meanNoLearn))
            f.write("StdNoLearn {}\n".format(self.stdNoLearn))
            
            
            f.write("PatientServiceLevel {}\n".format(self.alpha))
            f.write("AideServiceLevel {}\n".format(self.beta))
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("CallbackCall {}".format(nbcall))
            f.write('\n')
            f.write("LazyCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            #f.write("InfeasibleSubproblemsReachingTimeLimit {} \n".format(time_limit))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))
            
            
            #f.write("patient_id,aide_id,visit_time,assigned_aide_day_1,assigned_aide_day_2,assigned_aide_day_3,assigned_aide_day_4,assigned_aide_day_5")
            #f.write('\n')
            #for i in self.list_patient_new_id:
             #   f.write(str(self.instance.list_patient[i].id)+','+str(self.instance.list_patient[i].pre_assigned_aide[0])+
              #          ','+str(self.instance.list_patient[i].pre_assigned_aide[1])+','+
               #        str(self.instance.list_patient[i].pre_assigned_aide[2])+','+str(self.instance.list_patient[i].pre_assigned_aide[3])+','+
                #       str(self.instance.list_patient[i].pre_assigned_aide[4]))
                #f.write('\n')
                
        #"""


        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-TimeWindowRelaxation-route-EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-TimeWindowRelaxation-schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                


        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-TimeWindowRelaxation-Route-Schedule-EVT.txt"),"a") as f:

            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')
        
        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-TimeWindowRelaxation-Patient-CoC_EVT.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")

            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")

            for i in self.new_patient:
                if i.aide_assigned:
                    f.write('Patient {}'.format(i.id))
                    f.write('\n')
                    f.write('Aide(number of visits)')
                    f.write('\n')
                    f.write(str(i.aide_assigned))
                    f.write('\n')
                    f.write('\n')
            
            for i,j in self.frequency_visit.items():
                f.write('Actual number of visits {}: {} patients'.format(i,self.total_visit[i]))
                f.write('\n')
                f.write('Assigned Number of visits {}: {} patients'.format(i,j))
                f.write('\n')



def deterministic_no_learn_lazy_callback(model,where):
    
    if where==GRB.Callback.MIPSOL:

        #print("Current time: {}".format(grb_t))
        """
        if grb_t >= 240:
            model.terminate()
            return

        """

        model._call+=1
           
        yval=model.cbGetSolution(model._yvars)
        numval=model.cbGetSolution(model._numvars)
        
        
        list_Pik={k.id:{d:[i for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5] for d in 
                                 range(model._instance.horizon)} for k in model._activeAides}
        
        
        list_visit={k.id:{d:{i:int(np.round(numval[kk,i,dd])) for kk,i,dd in model._assignment.select(k.id,'*','*') if dd==d and yval[kk,i,dd]>0.5} for d in 
                                 range(model._instance.horizon)} for k in model._activeAides}
        

        
        #can_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        

        #print("the candidate objective is {}".format(model.cbGetSolution(model._objvar)))

    
        #print("the list of Pik is:")
        #print(list_Pik)
        
        Solve={k.id:0 for k in model._activeAides}
        
        for k in Solve:
            for d in range(model._instance.horizon):
                if len(list_Pik[k][d])>0:
                    Solve[k]=1
                    break
        for k in model._activeAides:
            
            if Solve[k.id]==1:
                #print("----------------------------------")
                #print("Solve Aide {}".format(k.id))
                logger.info("----------------------------------")
                logger.info("Solve Aide %s", k.id)

                #print("the list of Pik is:")
                #print(list_Pik[k.id])
                logger.info("the list of Pik %s", list_Pik[k.id])
                logger.info("the list of visit %s", list_visit[k.id])

                #print("the list of visit is:")

                #print(list_visit[k.id])

                """
                assignment_pattern=[]    
                for d in range(model._instance.horizon):
                    if list_Pik[k.id][d] not in assignment_pattern and list_Pik[k.id][d]:
                        assignment_pattern.append(list_Pik[k.id][d])
                        
                number_pattern=len(assignment_pattern)

                """
                value=""
                for d in range(model._instance.horizon):
                    if list_Pik[k.id][d]:
                        for b in list_Pik[k.id][d]:
                            value+=str(b)+'-'
                        value+="/"
                    
                    
                
                if value not in model._instance.list_feasible_per_aide[k.id]:
                    
                    #print("the assigned patients to aide {}".format(k.id+1))
                    #print(list_Pik[k.id])
                    subproblem=Deterministic_No_Learn_Subproblem(model._instance,k.id,list_Pik[k.id],list_visit[k.id],model._overtime,model._TCov,model._meanNoLearn)
                    model._subcall+=1
                    start=time.time()
                    over_time_status,status,solve_daily_sub,inf_daily_sub,complete_sub,dic_cut=subproblem.generate_cut(False)
                    end=time.time()
                    total_time=end-start
                    model._subTime+=total_time
                    model._complete_sub+=complete_sub
                    
                    if status==False:
                        
                        model._inf_complete_sub+=1   # increase the number of sub-problems solve completely and are infeasible
                        """
                        for d in range(model._instance.horizon):
                            if len(list_cut[d])>0:
                                model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i in list_cut[d])>=1)
                                model._nblazy+=1 
                        """
                        for d in dic_cut:
                            if len(dic_cut[d])>0:
                                model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i in dic_cut[d])>=1)
                                model._nblazy+=1
                        
                        #model.cbLazy(quicksum((1-model._yvars[k.id,i,d]) for i,d in list_cut)>=1)

                        
                        #print("the list of cut is:")
                        logger.info("the list of cut %s", dic_cut)
                        #print(dic_cut)
                        #print("----------------------------------")
                        logger.info("----------------------------------")

class Deterministic_No_Learn_Master_Problem(): # without learning effect, use only time window relaxation, deterministic subproblem 

    def __init__(self,file_name,instance,overtime,TCov,team,team_size,mean_no_learn):
        
        #print("Master Problem is created")
        
        self.instance=instance
        self.overtime=overtime
        self.TCov=TCov
        self.team_size=team_size
        self.file_name=file_name
        self.meanNoLearn=mean_no_learn

        
        
        
        
        #new patient are those who are available to be assigned to an aide
        self.new_patient=[instance.list_patient[i] for i in range(instance.number_patient) if instance.list_patient[i].
                            availability==2]
        
        
        #active aide are those who are available to serve patients
        self.active_aide=[instance.list_aide[i] for i in range(instance.number_aide) if instance.list_aide[i].
                            availability==2]
        
        self.list_aide_active_id=[]
        for k in self.active_aide:
            self.list_aide_active_id.append(k.id)

        self.list_patient_new_id=[]
        for i in self.new_patient:
            self.list_patient_new_id.append(i.id)

        #self.nb_patient=len(self.new_patient) 
        #self.nb_aide=len(self.active_aide)  
        

        self.horizon=instance.horizon
        
        if team:
            self.team_maker()
            
            #print("team maker is done")
            logger.info("team maker is done")

            #import pdb; pdb.set_trace()

            self.day_assignment=tuplelist([(k,i,d) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1])

            self.day_visit_frequency=tuplelist([(k,i,d,p) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i] for p in self.patient_visit_frequency[i][d][k]])


        else:
            self.day_assignment=tuplelist([(k.id,i.id,d) for k in self.active_aide for i in self.new_patient for d in range(self.horizon) if i.pre_assigned_day[d]==1])
        
        #to calculate maximum number of potential visits for each (k,i,d)
        
        
        
        #self.new_assignment=tuplelist([(k.id,i.id) for k in self.active_aide for i in self.free_patient])
        #print("the potential patient-aide assignment") 
        #print(self.new_assignment)
        #self.all_assignment=self.current_assignment+self.new_assignment
        #print("all assignments:")
        #print(self.all_assignment)
        
        self.morning_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[1]<=720]
        self.evening_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[0]>=720]

        #print("morning patient are: {}".format(self.morning_patient))
        #print("evening patient are: {}".format(self.evening_patient))
        logger.info("morning patient are: %s", self.morning_patient)
        logger.info("evening patient are: %s", self.evening_patient)

        #self.ugmentedDurationFrom={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_from[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}
        
        #self.ugmentedDurationTo={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_to[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}



        self.ugmentedDurationFrom={(k,i,d):(self.meanNoLearn[0])+self.min_travel_from[i][d][k] for k,i,d in self.day_assignment}

        self.ugmentedDurationTo={(k,i,d):(self.meanNoLearn[0])+self.min_travel_to[i][d][k] for k,i,d in self.day_assignment}

                   
                            
    
    """ It is a function to call the subproblem 
        when the master problem finds a feasible solution 
        and add feasibility cuts when they are required.
        It links the master and subproblem. 
    """
    
    def team_maker(self):
        random.seed(42)  # Set the seed once for reproducibility

        self.list_assignment={}
        for i in self.list_patient_new_id:

            self.list_assignment[i]=[]

            potential_aide=[]

            earliest_start=self.instance.list_patient[i].time_window[0]
            latest_start=self.instance.list_patient[i].time_window[1]-self.instance.list_patient[i].visit_duration
            visit_duration=self.instance.list_patient[i].visit_duration


            for k in self.list_aide_active_id:
                travel_time=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                time_window_start=self.instance.list_aide[k].time_window_start[0]
                time_window_end=self.instance.list_aide[k].time_window_end[1]

                if time_window_start + travel_time > latest_start:  # cannot reach the patient before his/her latest start time
                    continue
                elif travel_time+visit_duration > time_window_end + self.overtime - earliest_start:  
                    continue
                elif earliest_start + visit_duration +2.05 * (0.25 * visit_duration) > time_window_end + self.overtime:  
                    continue
                else:
                    potential_aide.append(k)

            if len(potential_aide)<=self.team_size:
                self.list_assignment[i]=potential_aide
            else:
                selected_aide=random.sample(potential_aide, self.team_size)
                self.list_assignment[i]=selected_aide

        self.patient_pattern={}

        for i in self.list_patient_new_id:
            self.patient_pattern[i]={}
            visit_days=[ d for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1]
            patterns=list(product(self.list_assignment[i], repeat=len(visit_days)))
            self.patient_pattern[i]={n:{day:p[d] for d,day in enumerate(visit_days)} for n,p in enumerate(patterns)}

        self.aide_day_patient={}
        for k in self.list_aide_active_id:
            self.aide_day_patient[k]={}
            for d in range(self.horizon):
                self.aide_day_patient[k][d]=[]
                for i in self.list_patient_new_id:
                    if self.instance.list_patient[i].pre_assigned_day[d]==1 and k in self.list_assignment[i]:
                        self.aide_day_patient[k][d].append(i)
        self.min_travel_from={}
        self.min_travel_to={}

        for i in self.list_patient_new_id:
            self.min_travel_from[i]={}
            self.min_travel_to[i]={}
            for d in range(self.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    self.min_travel_from[i][d]={}
                    self.min_travel_to[i][d]={}
                    for k in self.list_assignment[i]:
                        minTravelFrom=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]<minTravelFrom:
                                minTravelFrom=self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]
                        self.min_travel_from[i][d][k]=minTravelFrom

                        minTravelTo=100
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]<minTravelTo:
                                minTravelTo=self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]
                        if minTravelTo==100:
                            minTravelTo=0
                        self.min_travel_to[i][d][k]=minTravelTo
    
        
        self.patient_pattern_duration={}
        for i in self.list_patient_new_id:
            self.patient_pattern_duration[i]={}
            visit_duration=self.instance.list_patient[i].visit_duration
            for pattern_id,pattern in self.patient_pattern[i].items():
                self.patient_pattern_duration[i][pattern_id]={}
                aide_assignment_count={}
                for day,aide in pattern.items():
                    if aide not in aide_assignment_count:
                        aide_assignment_count[aide]=0
                    
                    aide_assignment_count[aide]+=1
                    learning_index=aide_assignment_count[aide]-1

                    self.patient_pattern_duration[i][pattern_id][day]=int(np.round(visit_duration))

        
        self.patient_visit_frequency={}
        self.patient_duration_frequency={}
        self.patient_day_frequency={}
        for i in self.list_patient_new_id:
            visit_duration=self.instance.list_patient[i].visit_duration
            self.patient_visit_frequency[i]={}
            self.patient_duration_frequency[i]={}
            self.patient_day_frequency[i]={}
            total_visit=0
            for d in range(self.instance.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    total_visit+=1
                    self.patient_duration_frequency[i][total_visit]=int(np.round((visit_duration)+1.64*(visit_duration*0.25)))
                    self.patient_visit_frequency[i][d]={}
                    self.patient_day_frequency[i][d]=[]
                    for k in self.list_assignment[i]:
                        self.patient_visit_frequency[i][d][k]=[]
                        for p in range(total_visit):
                            self.patient_visit_frequency[i][d][k].append(p+1)
                    for p in range(total_visit):
                        self.patient_day_frequency[i][d].append(p+1)
    
        
        self.patient_frequency_day={}

        for i in self.list_patient_new_id:
            self.patient_frequency_day[i]={}
            for p in self.patient_duration_frequency[i]:
                self.patient_frequency_day[i][p]=[]
                index=0
                for d in range(self.instance.horizon):
                    if self.instance.list_patient[i].pre_assigned_day[d]==1:
                        index+=1
                        if index>=p:
                            self.patient_frequency_day[i][p].append(d)
        
        """
        print("patient frequency day:")
        for i in self.list_patient_new_id:
            for p in self.patient_frequency_day[i]:
                print("patient {}, frequency {}, days {}".format(i,p,self.patient_frequency_day[i][p]))

        """


                    
        

            
                





            



        
    
    
    def branch_and_check(self,env):
        
        
            
        with Model(env=env) as m:

            #m=Model('Heching_master')
            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta") # whether a patient is acepted or not
            #x=m.addVars([(k, i) for i in self.list_patient_new_id for k in self.list_assignment[i]],vtype=GRB.BINARY, name="x")   # assignment of an aide to a patient over the horizon
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY, name="y") # daily assignment of an aide to a patient on requested days 
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0, ub=self.horizon,name="num") # the number of times an aide k has served patient i at end of day d
            #pattern=m.addVars([(i,pattern_id) for i in self.list_patient_new_id for pattern_id in self.patient_pattern[i]],vtype=GRB.BINARY,name="pattern") # whether a patient is served according to a specific pattern or not
            #freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq") # if aide k visited patient i on day d for the p th time

            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id), GRB.MAXIMIZE)  #self.instance.list_patient[i].number_visit


            
            #m.addConstr(obj<=self.maximum_mean_service_time+self.maximum_travel_time,"maximum_objective")
            #m.addConstr(obj<=self.maximum_obj,"maximum_objective")

            #m.addConstrs((quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i])==delta[i] for i in self.list_patient_new_id),"pattern_selection")
            #m.addConstrs((quicksum(x[k,i] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id),"driverCon")


            #m.addConstrs((y[k,i,d]==quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i] if d in self.patient_pattern[i][pattern_id] and self.patient_pattern[i][pattern_id][d]==k) for k,i,d in self.day_assignment),"aide_day_pattern")
            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            #m.addConstrs((y[k,i,d]<=x[k,i] for k,i,d in self.day_assignment),"unique_assigned_aide")


            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")


            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationFrom[kk,i,p,dd] for i in self.morning_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=72-self.instance.list_aide[kk].time_window_start[0] 
             #             for dd in range(self.horizon) for kk in self.list_aide_active_id),"morning_time")

            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationTo[kk,i,p,dd] for i in self.evening_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=self.instance.list_aide[kk].time_window_end[1]-72 
                      #    for dd in range(self.horizon) for kk in self.list_aide_active_id),"evening_time")


            m.addConstrs((quicksum(y[k,i,d]*(self.ugmentedDurationFrom[k,i,d]) for i in self.morning_patient if (k,i,d) in self.day_assignment)<=720-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")

            m.addConstrs((quicksum(y[k,i,d]*(self.ugmentedDurationTo[k,i,d]) for i in self.evening_patient if (k,i,d) in self.day_assignment)<=self.instance.list_aide[k].time_window_end[1]-720
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")

            """ comment all constraints related to frequency. No need to use for deterministic 
            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i]),"frequency_unique")
            
            m.addConstrs((quicksum((p)*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][p] if dd<=d )<= 
                          quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][p-1] if dd<d ) 
                          for i in self.list_patient_new_id for k in self.list_assignment[i] 
                          for p in self.patient_frequency_day[i] for d in self.patient_frequency_day[i][p] if p>1 ),"order_frequency")
            
            """

            #m.addConstrs((z[k,d]>=z[kk,d] for k,kk in self.dominance_pairs for d in range(self.horizon)),"dominance")

            #m.addConstrs((quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d1))<=quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d2)) for d1 in range(self.horizon) for d2 in range(d1+1,self.horizon) for kk in range(self.nb_aide)),"continuity")
            
            #m.addConstrs((y[k.id,i.id,d1]<=y[k.id,i.id,d2] for i in self.new_patient for k in self.active_aide for d1 in range(self.horizon) 
             #             for d2 in range(d1+1,self.horizon) if ((k.id,i.id,d1) in self.day_assignment) and ((k.id,i.id,d2) in self.day_assignment)),"continuity")

            
            #extra time_window constraints >>>it makes the solving time worse 

            """
            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_aide[k].time_window_start[0]+self.ugmentedDurationFrom[k,i,d]>self.instance.list_patient[i].time_window[1]),"time_window_morning")


            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_patient[i].time_window[0]+self.ugmentedDurationTo[k,i,d]>self.instance.list_aide[k].time_window_end[1]),"time_window_evening")

            m.addConstrs((quicksum(x[k,i]*(self.instance.list_patient[i].visit_duration)*(self.instance.list_patient[i].number_visit)
                                   for k,i in self.all_assignment.select(kk,'*'))<=self.instance.list_aide[kk].maximum_work_time for kk in range(self.nb_aide)),"over_time")

            """
            #m.write("heching-master.lp")self.instance.list_aide[k].time_window_end[0]


            m._yvars = y
            #m._freqvars = freq
            m._numvars = num
            m._instance=self.instance
            m._activeAides=self.active_aide
            m._assignment=self.day_assignment
            m._freqAssignment=self.day_visit_frequency
            m._call=0
            m._nblazy=0
            m._subcall=0
            m._subTime=0
            m._complete_sub=0
            m._inf_complete_sub=0
            m._time_limit=0
            #m._objvar=obj
            #m._scenario=create_scenario(self.instance,self.nb_scenario)
            m._overtime=self.overtime
            m._TCov=self.TCov
            m._meanNoLearn=self.meanNoLearn
            m.setParam("LazyConstraints",1)
            m.setParam("TimeLimit",14400)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            m.setParam("MIPFocus",1)
            m.setParam("Heuristics",0.3)
            #m.setParam("LogFile","log_file_master_approx.txt")
            #print(m.getParamInfo("TimeLimit"))
            #print(m.getAttr("NumVars"), m.getAttr("NumConstrs"))
            
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-EVT.lp")
            log_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic',  f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-EVT.log")
            m.setParam("LogFile", log_path)
            m.write(file_path)

            m.optimize(deterministic_no_learn_lazy_callback)
            if m.status == GRB.OPTIMAL:

                #print("the number of lazy constraint: {}".format(m._nblazy))
                #print("the run time is: {}".format(m.Runtime))
                #m.printAttr('X')
                #print("--------------------------------")
                logger.info("----------------------------------")
                #print("Optimal solution found with objective {}".format(m.getObjective().getValue()))
                logger.info("Optimal solution found with objective %s", m.getObjective().getValue())

                #print("--------------------------------")
                logger.info("----------------------------------")

                

                file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-solution-EVT.sol")
                m.write(file_path)
                f={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5] for d in 
                                         range(self.horizon)} for k in self.list_aide_active_id}

                list_visit={k:{d:{i:int(np.round(num[kk,i,dd].getAttr('X'))) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5} for d in
                                         range(self.horizon)} for k in self.list_aide_active_id}
                


                Solve={k.id:0 for k in self.active_aide}

                for k in Solve:
                    for d in range(self.instance.horizon):
                        if len(f[k][d])>0:
                            Solve[k]=1
                            break
                for k in self.active_aide:

                    if Solve[k.id]==1:
                        #print("Final Solve Aide {}".format(k.id))
                        #print("list Pik is:")
                        #print(f[k.id])
                        logger.info("Final Solve Aide %s", k.id)
                        logger.info("list Pik is: %s", f[k.id])
                        subproblem=Deterministic_No_Learn_Subproblem(self.instance,k.id,f[k.id],list_visit[k.id],self.overtime,self.TCov,self.meanNoLearn)
                        
                        subproblem.generate_cut(True)
                
                self.frequency_visit={i:0 for i in range(1,self.instance.horizon+1)}
                self.total_visit={i:0 for i in range(1,self.instance.horizon+1)}
                
                for i in self.new_patient:
                    if i.aide_assigned:

                        for z in i.aide_assigned.values():
                            self.frequency_visit[z]+=1
                        
                        self.total_visit[i.number_visit]+=1
                
                

                """
                for i in self.new_patient:
                    i.pre_assigned_aide=[-1 for d in range(self.horizon)]

                    for d in range(self.horizon):
                        if i.pre_assigned_day[d]==1:
                            for k in self.list_assignment[i]:
                                if y[k,i.id,d].getAttr('X')>0.5:
                                    i.pre_assigned_aide[d]=k
                                    break
                """
                self.write_solution(m.getObjective().getValue(),m._call,m._nblazy,m._subcall,m._complete_sub,m._inf_complete_sub,m._time_limit,m.Runtime,m._subTime,m.NodeCount)
                return m.Runtime,m.getObjective().getValue()
            elif m.status == GRB.TIME_LIMIT:
                #print("Time limit reached for master problem")
                logger.error("Time limit %s reached for master problem", m.Runtime)
                if m.SolCount>0:
                    #print("the best integer solution found has objective {}".format(m.ObjVal))
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    #print("the best bound is: {}".format(m.ObjBound))
                    logger.info("the best bound is: %s", m.ObjBound)
                    #print("the gap is: {}".format(m.MIPGap))
                    logger.info("the gap is: %s", m.MIPGap)
                else:
                    #print("No feasible solution found")
                    logger.warning("No feasible solution found")
                return m.Runtime, 0
            elif m.status ==GRB.INFEASIBLE:
                #print("The problem is infeasible")
                logger.error("The problem is infeasible")

                m.computeIIS()
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', 'Deterministic-infeasible.ilp')
                m.write(file_path)
                return 0, 0
            elif m.status== GRB.INTERRUPTED:
                #print("The optimization is interrupted")
                logger.error("The optimization is interrupted")
                if m.SolCount>0:
                    #print("the best integer solution found has objective {}".format(m.ObjVal))
                    #print("the best bound is: {}".format(m.ObjBound))
                    #print("the gap is: {}".format(m.MIPGap))
                    logger.info("the best integer solution found has objective %s", m.ObjVal)
                    logger.info("the best bound is: %s", m.ObjBound)
                    logger.info("the gap is: %s", m.MIPGap)
                else:
                    #print("No feasible solution found")
                    logger.warning("No feasible solution found")
                return m.Runtime, 0
                
    def write_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic')
        #"""
        with open(os.path.join(file_path,f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-output_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("TCov {}\n".format(self.TCov))
            f.write("MeanNoLearn {}\n".format(self.meanNoLearn))
            
            
    
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("CallbackCall {}".format(nbcall))
            f.write('\n')
            f.write("LazyCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            #f.write("InfeasibleSubproblemsReachingTimeLimit {} \n".format(time_limit))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))
            
            
            #f.write("patient_id,aide_id,visit_time,assigned_aide_day_1,assigned_aide_day_2,assigned_aide_day_3,assigned_aide_day_4,assigned_aide_day_5")
            #f.write('\n')
            #for i in self.list_patient_new_id:
             #   f.write(str(self.instance.list_patient[i].id)+','+str(self.instance.list_patient[i].pre_assigned_aide[0])+
              #          ','+str(self.instance.list_patient[i].pre_assigned_aide[1])+','+
               #        str(self.instance.list_patient[i].pre_assigned_aide[2])+','+str(self.instance.list_patient[i].pre_assigned_aide[3])+','+
                #       str(self.instance.list_patient[i].pre_assigned_aide[4]))
                #f.write('\n')
                
        #"""


        with open(os.path.join(file_path,f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-route-EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                


        with open(os.path.join(file_path,f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-Route-Schedule-EVT.txt"),"a") as f:

            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')
        
        with open(os.path.join(file_path,f"{self.file_name}-Deterministic-NoLearning-TimeWindowRelaxation-Patient-CoC_EVT.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")

            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and TimeWindowRelaxation EVT \n")

            for i in self.new_patient:
                if i.aide_assigned:
                    f.write('Patient {}'.format(i.id))
                    f.write('\n')
                    f.write('Aide(number of visits)')
                    f.write('\n')
                    f.write(str(i.aide_assigned))
                    f.write('\n')
                    f.write('\n')
            
            for i,j in self.frequency_visit.items():
                f.write('Actual number of visits {}: {} patients'.format(i,self.total_visit[i]))
                f.write('\n')
                f.write('Assigned Number of visits {}: {} patients'.format(i,j))
                f.write('\n')


class Assignment_Master_Problem(): # with learning and using assignment and time window relaxation
    
    def __init__(self,file_name,instance,overtime,alpha,beta,gamma,SCov,TCov, mean_learning_rate,std_learning_rate,team,team_size,similar_first):
        
        #print("Master Problem is created")
        
        self.instance=instance
        self.overtime=overtime
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.SCov=SCov
        self.TCov=TCov
        self.mean_learning=mean_learning_rate
        self.std_learning=std_learning_rate
        self.team_size=team_size
        self.similar_first=similar_first
        self.file_name=file_name
        
        
        
        
        #new patient are those who are available to be assigned to an aide
        self.new_patient=[instance.list_patient[i] for i in range(instance.number_patient) if instance.list_patient[i].
                            availability==2]
        
        
        #active aide are those who are available to serve patients
        self.active_aide=[instance.list_aide[i] for i in range(instance.number_aide) if instance.list_aide[i].
                            availability==2]
        
        self.list_aide_active_id=[]
        for k in self.active_aide:
            self.list_aide_active_id.append(k.id)

        self.list_patient_new_id=[]
        for i in self.new_patient:
            self.list_patient_new_id.append(i.id)

        #self.nb_patient=len(self.new_patient) 
        #self.nb_aide=len(self.active_aide)  
        

        self.horizon=instance.horizon

        self.morning_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[1]<=72]
        self.evening_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[0]>=72]

        print("morning patient are: {}".format(self.morning_patient))
        print("evening patient are: {}".format(self.evening_patient))
        
        if team:
            self.team_maker()
            print("team maker is done")

            #import pdb; pdb.set_trace()

            self.day_assignment=tuplelist([(k,i,d) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1])

            self.day_visit_frequency=tuplelist([(k,i,d,p) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i] for p in self.patient_visit_frequency[i][d][k]])

            self.route_assignment=tuplelist([(k,i,j,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d] for j in self.aide_day_patient[k][d] if i!=j and not (i in self.evening_patient and j in self.morning_patient)])

        else:
            self.day_assignment=tuplelist([(k.id,i.id,d) for k in self.active_aide for i in self.new_patient for d in range(self.horizon) if i.pre_assigned_day[d]==1])
        
        #to calculate maximum number of potential visits for each (k,i,d)
        
        self.num_day_assignment=[]
        for k,i,d in self.day_assignment:
            max_num_visit=sum([self.instance.list_patient[i].pre_assigned_day[dd] for dd in range(d+1)])
            for p in range(max_num_visit):
                self.num_day_assignment.append((k,i,d,p))

        
        self.learned_visit_duration={}
        self.non_learned_visit_duration={}
        for i in self.new_patient:
            p=0
            for d in range(self.horizon):
                if i.pre_assigned_day[d]==1:
                    self.learned_visit_duration[i.id,d]=int(np.round(i.visit_duration*self.mean_learning[p]))
                    self.non_learned_visit_duration[i.id,d]=i.visit_duration
                    p+=1
                
        self.minimum_mean_service_time=sum(self.learned_visit_duration.values())
        self.maximum_mean_service_time=sum(self.non_learned_visit_duration.values())

       
        self.minimum_travel_time=0
        self.maximum_travel_time=0          
        
        #self.new_assignment=tuplelist([(k.id,i.id) for k in self.active_aide for i in self.free_patient])
        #print("the potential patient-aide assignment") 
        #print(self.new_assignment)
        #self.all_assignment=self.current_assignment+self.new_assignment
        #print("all assignments:")
        #print(self.all_assignment)
        
        

        #self.ugmentedDurationFrom={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_from[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}
        
        #self.ugmentedDurationTo={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_to[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}


        
        self.ugmentedDurationFrom={(k,i,d,p):self.patient_duration_frequency[i][p] for k,i,d,p in self.day_visit_frequency}

        self.ugmentedDurationTo={(k,i,d,p):self.patient_duration_frequency[i][p] for k,i,d,p in self.day_visit_frequency}

        
                    
                            
    
    """ It is a function to call the subproblem 
        when the master problem finds a feasible solution 
        and add feasibility cuts when they are required.
        It links the master and subproblem. 
    """
    
    def team_maker(self):
        random.seed(42)  # Set the seed once for reproducibility

        self.list_assignment={}
        for i in self.list_patient_new_id:

            self.list_assignment[i]=[]

            potential_aide=[]

            earliest_start=self.instance.list_patient[i].time_window[0]
            latest_start=self.instance.list_patient[i].time_window[1]-self.instance.list_patient[i].visit_duration
            visit_duration=self.instance.list_patient[i].visit_duration


            for k in self.list_aide_active_id:
                travel_time=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                time_window_start=self.instance.list_aide[k].time_window_start[0]
                time_window_end=self.instance.list_aide[k].time_window_end[1]

                if time_window_start + travel_time > latest_start:  # cannot reach the patient before his/her latest start time
                    continue
                elif travel_time+visit_duration > time_window_end + self.overtime - earliest_start:  
                    continue
                elif earliest_start + visit_duration + self.beta * (self.SCov * visit_duration) > time_window_end + self.overtime:  
                    continue
                else:
                    potential_aide.append(k)

            if len(potential_aide)<=self.team_size:
                self.list_assignment[i]=potential_aide
            else:
                selected_aide=random.sample(potential_aide, self.team_size)
                self.list_assignment[i]=selected_aide

        self.patient_pattern={}

        for i in self.list_patient_new_id:
            self.patient_pattern[i]={}
            visit_days=[ d for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1]
            patterns=list(product(self.list_assignment[i], repeat=len(visit_days)))
            self.patient_pattern[i]={n:{day:p[d] for d,day in enumerate(visit_days)} for n,p in enumerate(patterns)}

        self.aide_day_patient={}
        for k in self.list_aide_active_id:
            #location_start=self.instance.list_aide[k].location_start + self.instance.number_node
            #location_end=self.instance.list_aide[k].location_end + self.instance.number_node

            self.aide_day_patient[k]={}
            for d in range(self.horizon):
                self.aide_day_patient[k][d]=[]
                for i in self.list_patient_new_id:
                    if self.instance.list_patient[i].pre_assigned_day[d]==1 and k in self.list_assignment[i]:
                        self.aide_day_patient[k][d].append(i)
                
        self.min_travel_from={}
        self.min_travel_to={}

        for i in self.list_patient_new_id:
            self.min_travel_from[i]={}
            self.min_travel_to[i]={}
            for d in range(self.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    self.min_travel_from[i][d]={}
                    self.min_travel_to[i][d]={}
                    for k in self.list_assignment[i]:
                        minTravelFrom=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]<minTravelFrom:
                                minTravelFrom=self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]
                        self.min_travel_from[i][d][k]=minTravelFrom

                        minTravelTo=100
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]<minTravelTo:
                                minTravelTo=self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]
                        if minTravelTo==100:
                            minTravelTo=0
                        
                        self.min_travel_to[i][d][k]=minTravelTo
    
        
        self.patient_pattern_duration={}
        for i in self.list_patient_new_id:
            self.patient_pattern_duration[i]={}
            visit_duration=self.instance.list_patient[i].visit_duration
            for pattern_id,pattern in self.patient_pattern[i].items():
                self.patient_pattern_duration[i][pattern_id]={}
                aide_assignment_count={}
                for day,aide in pattern.items():
                    if aide not in aide_assignment_count:
                        aide_assignment_count[aide]=0
                    
                    aide_assignment_count[aide]+=1
                    learning_index=aide_assignment_count[aide]-1

                    self.patient_pattern_duration[i][pattern_id][day]=int(np.round(visit_duration*self.mean_learning[learning_index]))

        
        self.patient_visit_frequency={}
        self.patient_duration_frequency={}
        self.patient_day_frequency={}
        for i in self.list_patient_new_id:
            visit_duration=self.instance.list_patient[i].visit_duration
            self.patient_visit_frequency[i]={}
            self.patient_duration_frequency[i]={}
            self.patient_day_frequency[i]={}
            total_visit=0
            for d in range(self.instance.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    total_visit+=1
                    self.patient_duration_frequency[i][total_visit]=(visit_duration*self.mean_learning[total_visit-1])+self.gamma*(visit_duration*self.SCov*self.std_learning[total_visit-1]) # +1*(visit_duration*self.SCov*self.std_learning[total_visit-1])
                    self.patient_visit_frequency[i][d]={}
                    self.patient_day_frequency[i][d]=[]
                    for k in self.list_assignment[i]:
                        self.patient_visit_frequency[i][d][k]=[]
                        for p in range(total_visit):
                            self.patient_visit_frequency[i][d][k].append(p+1)
                    for p in range(total_visit):
                        self.patient_day_frequency[i][d].append(p+1)
    
        
        self.patient_frequency_day={}

        for i in self.list_patient_new_id:
            self.patient_frequency_day[i]={}
            for p in self.patient_duration_frequency[i]:
                self.patient_frequency_day[i][p]=[]
                index=0
                for d in range(self.instance.horizon):
                    if self.instance.list_patient[i].pre_assigned_day[d]==1:
                        index+=1
                        if index>=p:
                            self.patient_frequency_day[i][p].append(d)
        
        """
        print("patient frequency day:")
        for i in self.list_patient_new_id:
            for p in self.patient_frequency_day[i]:
                print("patient {}, frequency {}, days {}".format(i,p,self.patient_frequency_day[i][p]))

        """


                    
        

            
                





            



        
    
    
    def branch_and_check(self,env):
        
        
            
        with Model(env=env) as m:

            #m=Model('Heching_master')
            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta") # whether a patient is acepted or not
            #x=m.addVars([(k, i) for i in self.list_patient_new_id for k in self.list_assignment[i]],vtype=GRB.BINARY, name="x")   # assignment of an aide to a patient over the horizon
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY, name="y") # daily assignment of an aide to a patient on requested days 
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0, ub=self.horizon,name="num") # the number of times an aide k has served patient i at end of day d
            #pattern=m.addVars([(i,pattern_id) for i in self.list_patient_new_id for pattern_id in self.patient_pattern[i]],vtype=GRB.BINARY,name="pattern") # whether a patient is served according to a specific pattern or not
            freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq") # if aide k visited patient i on day d for the p th time

            route=m.addVars(self.route_assignment,vtype=GRB.BINARY,name="route") # if aide k travels from i to j on day d

            route_from_start=m.addVars([(k,self.instance.list_aide[k].location_start,i,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d]],vtype=GRB.BINARY,name="route_from_start") # if aide k travels from start location to patient i on day d
            
            route_to_end=m.addVars([(k,i,self.instance.list_aide[k].location_end,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d]],vtype=GRB.BINARY,name="route_to_end") # if aide k travels from patient i to end location on day d
            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id), GRB.MAXIMIZE)  #self.instance.list_patient[i].number_visit

            m.addConstrs((route_from_start[k,self.instance.list_aide[k].location_start,i,d]+quicksum(route[k,j,i,d] for j in self.aide_day_patient[k][d] if j!=i and (k,j,i,d) in self.route_assignment)==y[k,i,d] 
                          for k,i,d in self.day_assignment),"route_from")

            m.addConstrs((route_to_end[k,i,self.instance.list_aide[k].location_end,d]+quicksum(route[k,i,j,d] for j in self.aide_day_patient[k][d] if j!=i and (k,i,j,d) in self.route_assignment)==y[k,i,d] 
                          for k,i,d in self.day_assignment),"route_to")

            m.addConstrs((route_from_start[k,self.instance.list_aide[k].location_start,i,d]+quicksum(route[k,j,i,d] for j in self.aide_day_patient[k][d] if j!=i and (k,j,i,d) in self.route_assignment)==
                         route_to_end[k,i,self.instance.list_aide[k].location_end,d]+quicksum(route[k,i,j,d] for j in self.aide_day_patient[k][d] if j!=i and (k,i,j,d) in self.route_assignment)
                         for k,i,d in self.day_assignment),"flow_conservation")
            
            m.addConstrs((quicksum(route_from_start[k,self.instance.list_aide[k].location_start,i,d] for i in self.aide_day_patient[k][d])<=1 for k in self.list_aide_active_id for d in range(self.horizon)),"start_route_limit")
            m.addConstrs((quicksum(route_to_end[k,i,self.instance.list_aide[k].location_end,d] for i in self.aide_day_patient[k][d])<=1 for k in self.list_aide_active_id for d in range(self.horizon)),"end_route_limit")
            
            #m.addConstr(obj<=self.maximum_mean_service_time+self.maximum_travel_time,"maximum_objective")
            #m.addConstr(obj<=self.maximum_obj,"maximum_objective")

            #m.addConstrs((quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i])==delta[i] for i in self.list_patient_new_id),"pattern_selection")
            #m.addConstrs((quicksum(x[k,i] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id),"driverCon")


            #m.addConstrs((y[k,i,d]==quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i] if d in self.patient_pattern[i][pattern_id] and self.patient_pattern[i][pattern_id][d]==k) for k,i,d in self.day_assignment),"aide_day_pattern")
            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            #m.addConstrs((y[k,i,d]<=x[k,i] for k,i,d in self.day_assignment),"unique_assigned_aide")


            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")


            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationFrom[kk,i,p,dd] for i in self.morning_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=72-self.instance.list_aide[kk].time_window_start[0] 
             #             for dd in range(self.horizon) for kk in self.list_aide_active_id),"morning_time")

            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationTo[kk,i,p,dd] for i in self.evening_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=self.instance.list_aide[kk].time_window_end[1]-72 
                      #    for dd in range(self.horizon) for kk in self.list_aide_active_id),"evening_time")

            
            m.addConstrs((quicksum(
                            quicksum(freq[k,i,d,p]*(self.ugmentedDurationFrom[k,i,d,p]) for p in self.patient_day_frequency[i][d])+
                            self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]*route_from_start[k,self.instance.list_aide[k].location_start,i,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]*route[k,j,i,d] for j in self.aide_day_patient[k][d] if i!=j and j in self.morning_patient)
                            for i in self.aide_day_patient[k][d] if i in self.morning_patient)
                          <=72-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")
            
            m.addConstrs((quicksum(
                            quicksum(freq[k,i,d,p]*(self.ugmentedDurationTo[k,i,d,p]) for p in self.patient_day_frequency[i][d])+
                            self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_aide[k].location_end]*route_to_end[k,i,self.instance.list_aide[k].location_end,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]*route[k,i,j,d] for j in self.aide_day_patient[k][d] if i!=j and j in self.evening_patient)
                            for i in self.aide_day_patient[k][d] if i in self.evening_patient)
                          <=self.instance.list_aide[k].time_window_end[1]-72
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")

            m.addConstrs((quicksum(
                            quicksum(freq[k,i,d,p]*(self.ugmentedDurationTo[k,i,d,p]) for p in self.patient_day_frequency[i][d])+
                            self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_aide[k].location_end]*route_to_end[k,i,self.instance.list_aide[k].location_end,d]+
                            self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]*route_from_start[k,self.instance.list_aide[k].location_start,i,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]*route[k,i,j,d] for j in self.aide_day_patient[k][d] if i!=j and (k,i,j,d) in self.route_assignment)
                            for i in self.aide_day_patient[k][d])
                          <=self.instance.list_aide[k].time_window_end[1]-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"aide_time_window")

            """
            m.addConstrs((quicksum(
                                    quicksum(freq[k,i,d,p]*(self.ugmentedDurationFrom[k,i,d,p]) for i in self.aide_day_patient[k][d] for p in self.patient_day_frequency[i][d])+
                                    quicksum(self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]*route_from_start[k,self.instance.list_aide[k].location_start,i,d] for i in self.aide_day_patient[k][d])+
                                    quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_aide[k].location_end]*route_to_end[k,i,self.instance.list_aide[k].location_end,d] for i in self.aide_day_patient[k][d])+
                                    quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]*route[k,i,j,d] for i in self.aide_day_patient[k][d] for j in self.aide_day_patient[k][d] if i!=j)
                                for d in range(self.horizon))
                                <=self.instance.list_aide[k].maximum_work_time 
                          for k in self.list_aide_active_id),"maximum_work_time")
            """

            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i]),"frequency_unique")
            
            m.addConstrs((quicksum((p)*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][p] if dd<=d )<= 
                          quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][p-1] if dd<d ) 
                          for i in self.list_patient_new_id for k in self.list_assignment[i] 
                          for p in self.patient_frequency_day[i] for d in self.patient_frequency_day[i][p] if p>1 ),"order_frequency")

            #m.addConstrs((z[k,d]>=z[kk,d] for k,kk in self.dominance_pairs for d in range(self.horizon)),"dominance")

            #m.addConstrs((quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d1))<=quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d2)) for d1 in range(self.horizon) for d2 in range(d1+1,self.horizon) for kk in range(self.nb_aide)),"continuity")
            
            #m.addConstrs((y[k.id,i.id,d1]<=y[k.id,i.id,d2] for i in self.new_patient for k in self.active_aide for d1 in range(self.horizon) 
             #             for d2 in range(d1+1,self.horizon) if ((k.id,i.id,d1) in self.day_assignment) and ((k.id,i.id,d2) in self.day_assignment)),"continuity")

            
            #extra time_window constraints >>>it makes the solving time worse 

            """
            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_aide[k].time_window_start[0]+self.ugmentedDurationFrom[k,i,d]>self.instance.list_patient[i].time_window[1]),"time_window_morning")


            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_patient[i].time_window[0]+self.ugmentedDurationTo[k,i,d]>self.instance.list_aide[k].time_window_end[1]),"time_window_evening")

            m.addConstrs((quicksum(x[k,i]*(self.instance.list_patient[i].visit_duration)*(self.instance.list_patient[i].number_visit)
                                   for k,i in self.all_assignment.select(kk,'*'))<=self.instance.list_aide[kk].maximum_work_time for kk in range(self.nb_aide)),"over_time")

            """
            #m.write("heching-master.lp")self.instance.list_aide[k].time_window_end[0]


            m._yvars = y
            m._freqvars = freq
            m._numvars = num
            m._instance=self.instance
            m._activeAides=self.active_aide
            m._assignment=self.day_assignment
            m._freqAssignment=self.day_visit_frequency
            m._call=0
            m._nblazy=0
            m._subcall=0
            m._subTime=0
            m._complete_sub=0
            m._inf_complete_sub=0
            m._time_limit=0
            #m._objvar=obj
            #m._scenario=create_scenario(self.instance,self.nb_scenario)
            m._overtime=self.overtime
            m._alpha=self.alpha
            m._beta=self.beta
            m._SCov=self.SCov
            m._TCov=self.TCov
            m._meanLearning=self.mean_learning
            m._stdLearning=self.std_learning
            m._similarFirst=self.similar_first
            m.setParam("LazyConstraints",1)
            m.setParam("TimeLimit",14400)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            #m.setParam("Heuristics",0)
            #m.setParam("LogFile","log_file_master_approx.txt")
            #print(m.getParamInfo("TimeLimit"))
            #print(m.getAttr("NumVars"), m.getAttr("NumConstrs"))
            
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path=os.path.join(self.BASE_DIR, 'result','learn',f"{self.file_name}-Learning-AssignmentRelaxation-EVT.lp")
            m.write(file_path)

            m.optimize(lazy_callback)
            if m.status == GRB.OPTIMAL:

                #print("the number of lazy constraint: {}".format(m._nblazy))
                #print("the run time is: {}".format(m.Runtime))
                #m.printAttr('X')
                print("--------------------------------")
                print("Optimal solution found with objective {}".format(m.getObjective().getValue()))

                print("--------------------------------")

                

                file_path=os.path.join(self.BASE_DIR, 'result','learn',f"{self.file_name}-Learning-AssignmentRelaxation-solution-approxEVT.sol")
                m.write(file_path)
                f={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5] for d in 
                                         range(self.horizon)} for k in self.list_aide_active_id}

                list_visit={k:{d:{i:int(np.round(num[kk,i,dd].getAttr('X'))) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5} for d in
                                         range(self.horizon)} for k in self.list_aide_active_id}
                list_Pik_frequency={k:{d:{i:p for kk,i,dd,p in self.day_visit_frequency.select(k,'*','*','*') if dd==d and freq[kk,i,dd,p].getAttr('X')>0.5} for d in
                                         range(self.horizon)} for k in self.list_aide_active_id}



                Solve={k.id:0 for k in self.active_aide}

                for k in Solve:
                    for d in range(self.instance.horizon):
                        if len(f[k][d])>0:
                            Solve[k]=1
                            break
                for k in self.active_aide:

                    if Solve[k.id]==1:
                        print("Final Solve Aide {}".format(k.id))
                        print("list Pik is:")
                        print(f[k.id])
                        subproblem=Subproblem(self.instance,k.id,f[k.id],list_visit[k.id],list_Pik_frequency[k.id],self.overtime,self.alpha,self.beta,self.SCov,self.TCov,self.mean_learning,self.std_learning)
                        
                        subproblem.generate_cut(True,False)

                """
                for i in self.new_patient:
                    i.pre_assigned_aide=[-1 for d in range(self.horizon)]

                    for d in range(self.horizon):
                        if i.pre_assigned_day[d]==1:
                            for k in self.list_assignment[i]:
                                if y[k,i.id,d].getAttr('X')>0.5:
                                    i.pre_assigned_aide[d]=k
                                    break
                """
                self.write_solution(m.getObjective().getValue(),m._call,m._nblazy,m._subcall,m._complete_sub,m._inf_complete_sub,m._time_limit,m.Runtime,m._subTime,m.NodeCount)
                return m.Runtime,m.getObjective().getValue()
            elif m.status == GRB.TIME_LIMIT:
                print("Time limit reached for master problem")
                if m.SolCount>0:
                    print("the best integer solution found has objective {}".format(m.ObjVal))
                    print("the best bound is: {}".format(m.ObjBound))
                    print("the gap is: {}".format(m.MIPGap))
                else:
                    print("No feasible solution found")
                return m.Runtime, 0
            elif m.status ==GRB.INFEASIBLE:
                print("The problem is infeasible")

                m.computeIIS()
                file_path=os.path.join(self.BASE_DIR, 'result','learn','infeasible.ilp')
                m.write(file_path)
                return 0, 0
            elif m.status== GRB.INTERRUPTED:
                print("The optimization is interrupted")
                if m.SolCount>0:
                    print("the best integer solution found has objective {}".format(m.ObjVal))
                    print("the best bound is: {}".format(m.ObjBound))
                    print("the gap is: {}".format(m.MIPGap))
                else:
                    print("No feasible solution found")
                return m.Runtime, 0
                
                        
                                    
                        
        
        
        
    
    def write_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','learn')
        #"""
        with open(os.path.join(file_path,f"{self.file_name}-Learning-AssignmentRelaxation-output_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and AssignmentRelaxation \n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("SCov {}\n".format(self.SCov))
            f.write("TCov {}\n".format(self.TCov))
            f.write("MeanLearningRate {}\n".format(self.mean_learning))
            f.write("StdLearningRate {}\n".format(self.std_learning))
            
            f.write("PatientServiceLevel {}\n".format(self.alpha))
            f.write("AideServiceLevel {}\n".format(self.beta))
            f.write("Gamma {}\n".format(self.gamma))
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("CallbackCall {}".format(nbcall))
            f.write('\n')
            f.write("LazyCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            #f.write("InfeasibleSubproblemsReachingTimeLimit {} \n".format(time_limit))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))
            
            
            #f.write("patient_id,aide_id,visit_time,assigned_aide_day_1,assigned_aide_day_2,assigned_aide_day_3,assigned_aide_day_4,assigned_aide_day_5")
            #f.write('\n')
            #for i in self.list_patient_new_id:
             #   f.write(str(self.instance.list_patient[i].id)+','+str(self.instance.list_patient[i].pre_assigned_aide[0])+
              #          ','+str(self.instance.list_patient[i].pre_assigned_aide[1])+','+
               #        str(self.instance.list_patient[i].pre_assigned_aide[2])+','+str(self.instance.list_patient[i].pre_assigned_aide[3])+','+
                #       str(self.instance.list_patient[i].pre_assigned_aide[4]))
                #f.write('\n')
                
        #"""
            
            
        with open(os.path.join(file_path,f"{self.file_name}-Learning-AssignmentRelaxation-route_approx-EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')
        
        with open(os.path.join(file_path,f"{self.file_name}-Learning-AssignmentRelaxation-schedule_EVT.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                


        with open(os.path.join(file_path,f"{self.file_name}-Learning-AssignmentRelaxation-Route-Schedule_EVT.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method Learning and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')
                    
                    

            
        
        #"""
    

class Assignment_No_Learn_Master_Problem(): # without learning effect and using assignment and time window relaxation 

    def __init__(self,file_name,instance,overtime,alpha,beta,gamma,SCov,TCov,team,team_size):
        
        #print("Master Problem is created")
        
        self.instance=instance
        self.overtime=overtime
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.SCov=SCov
        self.TCov=TCov
        self.team_size=team_size
        self.file_name=file_name

        
        
        
        
        #new patient are those who are available to be assigned to an aide
        self.new_patient=[instance.list_patient[i] for i in range(instance.number_patient) if instance.list_patient[i].
                            availability==2]
        
        
        #active aide are those who are available to serve patients
        self.active_aide=[instance.list_aide[i] for i in range(instance.number_aide) if instance.list_aide[i].
                            availability==2]
        
        self.list_aide_active_id=[]
        for k in self.active_aide:
            self.list_aide_active_id.append(k.id)

        self.list_patient_new_id=[]
        for i in self.new_patient:
            self.list_patient_new_id.append(i.id)

        #self.nb_patient=len(self.new_patient) 
        #self.nb_aide=len(self.active_aide)  
        

        self.horizon=instance.horizon

        self.morning_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[1]<=72]
        self.evening_patient=[i for i in self.list_patient_new_id if self.instance.list_patient[i].time_window[0]>=72]

        print("morning patient are: {}".format(self.morning_patient))
        print("evening patient are: {}".format(self.evening_patient))

        
        if team:
            self.team_maker()
            print("team maker is done")

            #import pdb; pdb.set_trace()

            self.day_assignment=tuplelist([(k,i,d) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1])

            self.day_visit_frequency=tuplelist([(k,i,d,p) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i] for p in self.patient_visit_frequency[i][d][k]])
            
            
            self.route_assignment=tuplelist([(k,i,j,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d] for j in self.aide_day_patient[k][d] if i!=j and not(i in self.evening_patient and j in self.morning_patient)])


        else:
            self.day_assignment=tuplelist([(k.id,i.id,d) for k in self.active_aide for i in self.new_patient for d in range(self.horizon) if i.pre_assigned_day[d]==1])
        
        #to calculate maximum number of potential visits for each (k,i,d)
        
        
        
        #self.new_assignment=tuplelist([(k.id,i.id) for k in self.active_aide for i in self.free_patient])
        #print("the potential patient-aide assignment") 
        #print(self.new_assignment)
        #self.all_assignment=self.current_assignment+self.new_assignment
        #print("all assignments:")
        #print(self.all_assignment)
        
        
        #self.ugmentedDurationFrom={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_from[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}
        
        #self.ugmentedDurationTo={(k,i,p,d):self.patient_pattern_duration[i][p][d]+self.min_travel_to[i][d][k] for i in self.list_patient_new_id for p in self.patient_pattern[i] for d,k in self.patient_pattern[i][p].items()}



        self.ugmentedDurationFrom={(k,i,d):self.instance.list_patient[i].visit_duration*(1+self.gamma*self.SCov) for k,i,d in self.day_assignment}

        self.ugmentedDurationTo={(k,i,d):self.instance.list_patient[i].visit_duration*(1+self.gamma*self.SCov) for k,i,d in self.day_assignment}

        
                    
                            
    
    """ It is a function to call the subproblem 
        when the master problem finds a feasible solution 
        and add feasibility cuts when they are required.
        It links the master and subproblem. 
    """
    
    def team_maker(self):
        random.seed(42)  # Set the seed once for reproducibility

        self.list_assignment={}
        for i in self.list_patient_new_id:

            self.list_assignment[i]=[]

            potential_aide=[]

            earliest_start=self.instance.list_patient[i].time_window[0]
            latest_start=self.instance.list_patient[i].time_window[1]-self.instance.list_patient[i].visit_duration
            visit_duration=self.instance.list_patient[i].visit_duration


            for k in self.list_aide_active_id:
                travel_time=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                time_window_start=self.instance.list_aide[k].time_window_start[0]
                time_window_end=self.instance.list_aide[k].time_window_end[1]

                if time_window_start + travel_time > latest_start:  # cannot reach the patient before his/her latest start time
                    continue
                elif travel_time+visit_duration > time_window_end + self.overtime - earliest_start:  
                    continue
                elif earliest_start + visit_duration + self.beta * (self.SCov * visit_duration) > time_window_end + self.overtime:  
                    continue
                else:
                    potential_aide.append(k)

            if len(potential_aide)<=self.team_size:
                self.list_assignment[i]=potential_aide
            else:
                selected_aide=random.sample(potential_aide, self.team_size)
                self.list_assignment[i]=selected_aide

        self.patient_pattern={}

        for i in self.list_patient_new_id:
            self.patient_pattern[i]={}
            visit_days=[ d for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1]
            patterns=list(product(self.list_assignment[i], repeat=len(visit_days)))
            self.patient_pattern[i]={n:{day:p[d] for d,day in enumerate(visit_days)} for n,p in enumerate(patterns)}

        self.aide_day_patient={}
        for k in self.list_aide_active_id:
            self.aide_day_patient[k]={}
            for d in range(self.horizon):
                self.aide_day_patient[k][d]=[]
                for i in self.list_patient_new_id:
                    if self.instance.list_patient[i].pre_assigned_day[d]==1 and k in self.list_assignment[i]:
                        self.aide_day_patient[k][d].append(i)
        self.min_travel_from={}
        self.min_travel_to={}

        for i in self.list_patient_new_id:
            self.min_travel_from[i]={}
            self.min_travel_to[i]={}
            for d in range(self.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    self.min_travel_from[i][d]={}
                    self.min_travel_to[i][d]={}
                    for k in self.list_assignment[i]:
                        minTravelFrom=self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]<minTravelFrom:
                                minTravelFrom=self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]
                        self.min_travel_from[i][d][k]=minTravelFrom

                        minTravelTo=100
                        for j in self.aide_day_patient[k][d]:
                            if j!=i and self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]<minTravelTo:
                                minTravelTo=self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]
                        
                        if minTravelTo==100:
                            minTravelTo=0
                        
                        self.min_travel_to[i][d][k]=minTravelTo
    
        
        self.patient_pattern_duration={}
        for i in self.list_patient_new_id:
            self.patient_pattern_duration[i]={}
            visit_duration=self.instance.list_patient[i].visit_duration
            for pattern_id,pattern in self.patient_pattern[i].items():
                self.patient_pattern_duration[i][pattern_id]={}
                aide_assignment_count={}
                for day,aide in pattern.items():
                    if aide not in aide_assignment_count:
                        aide_assignment_count[aide]=0
                    
                    aide_assignment_count[aide]+=1
                    learning_index=aide_assignment_count[aide]-1

                    self.patient_pattern_duration[i][pattern_id][day]=int(np.round(visit_duration))

        
        self.patient_visit_frequency={}
        self.patient_duration_frequency={}
        self.patient_day_frequency={}
        for i in self.list_patient_new_id:
            visit_duration=self.instance.list_patient[i].visit_duration
            self.patient_visit_frequency[i]={}
            self.patient_duration_frequency[i]={}
            self.patient_day_frequency[i]={}
            total_visit=0
            for d in range(self.instance.horizon):
                if self.instance.list_patient[i].pre_assigned_day[d]==1:
                    total_visit+=1
                    self.patient_duration_frequency[i][total_visit]=int(np.round((visit_duration)+1.64*(visit_duration*self.SCov)))
                    self.patient_visit_frequency[i][d]={}
                    self.patient_day_frequency[i][d]=[]
                    for k in self.list_assignment[i]:
                        self.patient_visit_frequency[i][d][k]=[]
                        for p in range(total_visit):
                            self.patient_visit_frequency[i][d][k].append(p+1)
                    for p in range(total_visit):
                        self.patient_day_frequency[i][d].append(p+1)
    
        
        self.patient_frequency_day={}

        for i in self.list_patient_new_id:
            self.patient_frequency_day[i]={}
            for p in self.patient_duration_frequency[i]:
                self.patient_frequency_day[i][p]=[]
                index=0
                for d in range(self.instance.horizon):
                    if self.instance.list_patient[i].pre_assigned_day[d]==1:
                        index+=1
                        if index>=p:
                            self.patient_frequency_day[i][p].append(d)
        
        """
        print("patient frequency day:")
        for i in self.list_patient_new_id:
            for p in self.patient_frequency_day[i]:
                print("patient {}, frequency {}, days {}".format(i,p,self.patient_frequency_day[i][p]))

        """


                    
        

            
                





            



        
    
    
    def branch_and_check(self,env):
        
        
            
        with Model(env=env) as m:

            #m=Model('Heching_master')
            delta=m.addVars(self.list_patient_new_id,vtype=GRB.BINARY,name="delta") # whether a patient is acepted or not
            #x=m.addVars([(k, i) for i in self.list_patient_new_id for k in self.list_assignment[i]],vtype=GRB.BINARY, name="x")   # assignment of an aide to a patient over the horizon
            y=m.addVars(self.day_assignment,vtype=GRB.BINARY, name="y") # daily assignment of an aide to a patient on requested days 
            num=m.addVars(self.day_assignment,vtype=GRB.INTEGER,lb=0, ub=self.horizon,name="num") # the number of times an aide k has served patient i at end of day d
            #pattern=m.addVars([(i,pattern_id) for i in self.list_patient_new_id for pattern_id in self.patient_pattern[i]],vtype=GRB.BINARY,name="pattern") # whether a patient is served according to a specific pattern or not
            freq=m.addVars(self.day_visit_frequency,vtype=GRB.BINARY,name="freq") # if aide k visited patient i on day d for the p th time

            route=m.addVars(self.route_assignment,vtype=GRB.BINARY,name="route") # if aide k travels from i to j on day d

            route_from_start=m.addVars([(k,self.instance.list_aide[k].location_start,i,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d]],vtype=GRB.BINARY,name="route_from_start") # if aide k travels from start location to patient i on day d
            
            route_to_end=m.addVars([(k,i,self.instance.list_aide[k].location_end,d) for k in self.list_aide_active_id for d in range(self.horizon) for i in self.aide_day_patient[k][d]],vtype=GRB.BINARY,name="route_to_end") # if aide k travels from patient i to end location on day d



            m.setObjective(quicksum(self.instance.list_patient[i].number_visit*delta[i] for i in self.list_patient_new_id), GRB.MAXIMIZE)  #self.instance.list_patient[i].number_visit

            m.addConstrs((route_from_start[k,self.instance.list_aide[k].location_start,i,d]+quicksum(route[k,j,i,d] for j in self.aide_day_patient[k][d] if j!=i and (k,j,i,d) in self.route_assignment)==y[k,i,d] 
                          for k,i,d in self.day_assignment),"route_from")

            m.addConstrs((route_to_end[k,i,self.instance.list_aide[k].location_end,d]+quicksum(route[k,i,j,d] for j in self.aide_day_patient[k][d] if j!=i and (k,i,j,d) in self.route_assignment)==y[k,i,d] 
                          for k,i,d in self.day_assignment),"route_to")
            
            m.addConstrs((route_from_start[k,self.instance.list_aide[k].location_start,i,d]+quicksum(route[k,j,i,d] for j in self.aide_day_patient[k][d] if j!=i and (k,j,i,d) in self.route_assignment)==
                         route_to_end[k,i,self.instance.list_aide[k].location_end,d]+quicksum(route[k,i,j,d] for j in self.aide_day_patient[k][d] if j!=i and (k,i,j,d) in self.route_assignment)
                         for k,i,d in self.day_assignment),"flow_conservation")
            
            m.addConstrs((quicksum(route_from_start[k,self.instance.list_aide[k].location_start,i,d] for i in self.aide_day_patient[k][d])<=1 for k in self.list_aide_active_id for d in range(self.horizon)),"start_route_limit")
            m.addConstrs((quicksum(route_to_end[k,i,self.instance.list_aide[k].location_end,d] for i in self.aide_day_patient[k][d])<=1 for k in self.list_aide_active_id for d in range(self.horizon)),"end_route_limit")
            
            

            #m.addConstr(obj<=self.maximum_mean_service_time+self.maximum_travel_time,"maximum_objective")
            #m.addConstr(obj<=self.maximum_obj,"maximum_objective")

            #m.addConstrs((quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i])==delta[i] for i in self.list_patient_new_id),"pattern_selection")
            #m.addConstrs((quicksum(x[k,i] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id),"driverCon")


            #m.addConstrs((y[k,i,d]==quicksum(pattern[i,pattern_id] for pattern_id in self.patient_pattern[i] if d in self.patient_pattern[i][pattern_id] and self.patient_pattern[i][pattern_id][d]==k) for k,i,d in self.day_assignment),"aide_day_pattern")
            m.addConstrs((quicksum(y[k,i,d] for k in self.list_assignment[i])==delta[i] for i in self.list_patient_new_id for d in range(self.horizon) if self.instance.list_patient[i].pre_assigned_day[d]==1),"requested_visit")

            #m.addConstrs((y[k,i,d]<=x[k,i] for k,i,d in self.day_assignment),"unique_assigned_aide")


            m.addConstrs((quicksum(y[k,i,dd] for dd in range(0,d+1) if (k,i,dd) in self.day_assignment)==num[k,i,d] for k,i,d in self.day_assignment),"number_visit")


            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationFrom[kk,i,p,dd] for i in self.morning_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=72-self.instance.list_aide[kk].time_window_start[0] 
             #             for dd in range(self.horizon) for kk in self.list_aide_active_id),"morning_time")

            #m.addConstrs((quicksum(pattern[i,p]*self.ugmentedDurationTo[kk,i,p,dd] for i in self.evening_patient for p in self.patient_pattern[i] if dd in self.patient_pattern[i][p] and self.patient_pattern[i][p][dd]==kk)<=self.instance.list_aide[kk].time_window_end[1]-72 
                      #    for dd in range(self.horizon) for kk in self.list_aide_active_id),"evening_time")


            m.addConstrs((quicksum(
                            y[k,i,d]*(self.ugmentedDurationFrom[k,i,d]) +
                            self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]*route_from_start[k,self.instance.list_aide[k].location_start,i,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[j].location][self.instance.list_patient[i].location]*route[k,j,i,d] for j in self.aide_day_patient[k][d] if i!=j and j in self.morning_patient)
                            for i in self.aide_day_patient[k][d] if i in self.morning_patient)
                          <=72-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"morning_time")
            
            m.addConstrs((quicksum(
                            y[k,i,d]*(self.ugmentedDurationTo[k,i,d])+
                            self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_aide[k].location_end]*route_to_end[k,i,self.instance.list_aide[k].location_end,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]*route[k,i,j,d] for j in self.aide_day_patient[k][d] if i!=j if j in self.evening_patient)
                            for i in self.aide_day_patient[k][d] if i in self.evening_patient)
                          <=self.instance.list_aide[k].time_window_end[1]-72
                          for d in range(self.horizon) for k in self.list_aide_active_id),"evening_time")

            m.addConstrs((quicksum(
                            y[k,i,d]*(self.ugmentedDurationTo[k,i,d])+
                            self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_aide[k].location_end]*route_to_end[k,i,self.instance.list_aide[k].location_end,d]+
                            self.instance.distance_matrix[self.instance.list_aide[k].location_start][self.instance.list_patient[i].location]*route_from_start[k,self.instance.list_aide[k].location_start,i,d]+
                            quicksum(self.instance.distance_matrix[self.instance.list_patient[i].location][self.instance.list_patient[j].location]*route[k,i,j,d] for j in self.aide_day_patient[k][d] if i!=j and (k,i,j,d) in self.route_assignment)
                            for i in self.aide_day_patient[k][d])
                          <=self.instance.list_aide[k].time_window_end[1]-self.instance.list_aide[k].time_window_start[0]
                          for d in range(self.horizon) for k in self.list_aide_active_id),"aide_time_window")

            
            m.addConstrs((quicksum(freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])==y[k,i,d] for k,i,d in self.day_assignment),"frequency_definition")

            m.addConstrs((quicksum(freq[k,i,d,p] for d in self.patient_frequency_day[i][p])<=1 for i in self.list_patient_new_id for k in self.list_assignment[i] for p in self.patient_frequency_day[i]),"frequency_unique")
            
            m.addConstrs((quicksum((p)*freq[k,i,d,p] for p in self.patient_visit_frequency[i][d][k])<=quicksum(y[k,i,dd] for dd in range(d+1) if (k,i,dd) in self.day_assignment) for i in self.list_patient_new_id for k in self.list_assignment[i] for d in self.patient_visit_frequency[i]),"cumulative_frequency")

            m.addConstrs((quicksum(freq[k,i,dd,p] for dd in self.patient_frequency_day[i][p] if dd<=d )<= 
                          quicksum(freq[k,i,dd,p-1] for dd in self.patient_frequency_day[i][p-1] if dd<d ) 
                          for i in self.list_patient_new_id for k in self.list_assignment[i] 
                          for p in self.patient_frequency_day[i] for d in self.patient_frequency_day[i][p] if p>1 ),"order_frequency")

            #m.addConstrs((z[k,d]>=z[kk,d] for k,kk in self.dominance_pairs for d in range(self.horizon)),"dominance")

            #m.addConstrs((quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d1))<=quicksum(y[k,i,d] for k,i,d in self.day_assignment.select(kk,'*',d2)) for d1 in range(self.horizon) for d2 in range(d1+1,self.horizon) for kk in range(self.nb_aide)),"continuity")
            
            #m.addConstrs((y[k.id,i.id,d1]<=y[k.id,i.id,d2] for i in self.new_patient for k in self.active_aide for d1 in range(self.horizon) 
             #             for d2 in range(d1+1,self.horizon) if ((k.id,i.id,d1) in self.day_assignment) and ((k.id,i.id,d2) in self.day_assignment)),"continuity")

            
            #extra time_window constraints >>>it makes the solving time worse 

            """
            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_aide[k].time_window_start[0]+self.ugmentedDurationFrom[k,i,d]>self.instance.list_patient[i].time_window[1]),"time_window_morning")


            m.addConstrs((y[k,i,d]==0 for d in range(self.horizon) for k,i in self.all_assignment 
                          if self.instance.list_patient[i].time_window[0]+self.ugmentedDurationTo[k,i,d]>self.instance.list_aide[k].time_window_end[1]),"time_window_evening")

            m.addConstrs((quicksum(x[k,i]*(self.instance.list_patient[i].visit_duration)*(self.instance.list_patient[i].number_visit)
                                   for k,i in self.all_assignment.select(kk,'*'))<=self.instance.list_aide[kk].maximum_work_time for kk in range(self.nb_aide)),"over_time")

            """
            #m.write("heching-master.lp")self.instance.list_aide[k].time_window_end[0]


            m._yvars = y
            m._freqvars = freq
            m._numvars = num
            m._instance=self.instance
            m._activeAides=self.active_aide
            m._assignment=self.day_assignment
            m._freqAssignment=self.day_visit_frequency
            m._call=0
            m._nblazy=0
            m._subcall=0
            m._subTime=0
            m._complete_sub=0
            m._inf_complete_sub=0
            m._time_limit=0
            #m._objvar=obj
            #m._scenario=create_scenario(self.instance,self.nb_scenario)
            m._overtime=self.overtime
            m._alpha=self.alpha
            m._beta=self.beta
            m._SCov=self.SCov
            m._TCov=self.TCov
            m.setParam("LazyConstraints",1)
            m.setParam("TimeLimit",14400)
            m.setParam("Presolve",0)
            m.setParam("Threads",1)
            #m.setParam("Heuristics",0)
            #m.setParam("LogFile","log_file_master_approx.txt")
            #print(m.getParamInfo("TimeLimit"))
            #print(m.getAttr("NumVars"), m.getAttr("NumConstrs"))
            
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path=os.path.join(self.BASE_DIR, 'result','no-learn',f"{self.file_name}-NoLearning-AssignmentRelaxation-EVT.lp")
            m.write(file_path)

            m.optimize(no_learn_lazy_callback)
            if m.status == GRB.OPTIMAL:

                #print("the number of lazy constraint: {}".format(m._nblazy))
                #print("the run time is: {}".format(m.Runtime))
                #m.printAttr('X')
                print("--------------------------------")
                print("Optimal solution found with objective {}".format(m.getObjective().getValue()))

                print("--------------------------------")

                

                file_path=os.path.join(self.BASE_DIR, 'result','no-learn',f"{self.file_name}-NoLearning-AssignmentRelaxation-solution-EVT.sol")
                m.write(file_path)
                f={k:{d:[i for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5] for d in 
                                         range(self.horizon)} for k in self.list_aide_active_id}

                list_visit={k:{d:{i:int(np.round(num[kk,i,dd].getAttr('X'))) for kk,i,dd in self.day_assignment.select(k,'*','*') if dd==d and y[kk,i,dd].getAttr('X')>0.5} for d in
                                         range(self.horizon)} for k in self.list_aide_active_id}
                


                Solve={k.id:0 for k in self.active_aide}

                for k in Solve:
                    for d in range(self.instance.horizon):
                        if len(f[k][d])>0:
                            Solve[k]=1
                            break
                for k in self.active_aide:

                    if Solve[k.id]==1:
                        print("Final Solve Aide {}".format(k.id))
                        print("list Pik is:")
                        print(f[k.id])
                        subproblem=No_Learn_Subproblem(self.instance,k.id,f[k.id],list_visit[k.id],self.overtime,self.alpha,self.beta,self.SCov,self.TCov)
                        
                        subproblem.generate_cut(True)

                """
                for i in self.new_patient:
                    i.pre_assigned_aide=[-1 for d in range(self.horizon)]

                    for d in range(self.horizon):
                        if i.pre_assigned_day[d]==1:
                            for k in self.list_assignment[i]:
                                if y[k,i.id,d].getAttr('X')>0.5:
                                    i.pre_assigned_aide[d]=k
                                    break
                """
                self.write_solution(m.getObjective().getValue(),m._call,m._nblazy,m._subcall,m._complete_sub,m._inf_complete_sub,m._time_limit,m.Runtime,m._subTime,m.NodeCount)
                return m.Runtime,m.getObjective().getValue()
            elif m.status == GRB.TIME_LIMIT:
                print("Time limit reached for master problem")
                if m.SolCount>0:
                    print("the best integer solution found has objective {}".format(m.ObjVal))
                    print("the best bound is: {}".format(m.ObjBound))
                    print("the gap is: {}".format(m.MIPGap))
                else:
                    print("No feasible solution found")
                return m.Runtime, 0
            elif m.status ==GRB.INFEASIBLE:
                print("The problem is infeasible")

                m.computeIIS()
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn','infeasible.ilp')
                m.write(file_path)
                return 0, 0
            elif m.status== GRB.INTERRUPTED:
                print("The optimization is interrupted")
                if m.SolCount>0:
                    print("the best integer solution found has objective {}".format(m.ObjVal))
                    print("the best bound is: {}".format(m.ObjBound))
                    print("the gap is: {}".format(m.MIPGap))
                else:
                    print("No feasible solution found")
                return m.Runtime, 0
                
    def write_solution(self,obj,nbcall,nblazy,subcall,complete_sub,inf_sub,time_limit,comp_time,sub_time,node_count):

        file_path=os.path.join(self.BASE_DIR, 'result','no-learn')
        #"""
        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-AssignmentRelaxation-output_approx.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and AssignmentRelaxation\n")
            f.write("Patients {}\n".format(len(self.new_patient)))
            f.write("ActiveAides {}\n".format(len(self.active_aide)))
            f.write("SCov {}\n".format(self.SCov))
            f.write("TCov {}\n".format(self.TCov))
            
            
            f.write("PatientServiceLevel {}\n".format(self.alpha))
            f.write("AideServiceLevel {}\n".format(self.beta))
            f.write("ObjectiveValue {} \n".format(obj))
            f.write("CallbackCall {}".format(nbcall))
            f.write('\n')
            f.write("LazyCut {}".format(nblazy))
            f.write('\n')
            f.write("SubproblemCall {}".format(subcall))
            f.write('\n')
            f.write("SubproblemsSolvedCompletely {} \n".format(complete_sub))
            f.write("CompleteSubproblemsInfeasible {} \n".format(inf_sub))
            #f.write("InfeasibleSubproblemsReachingTimeLimit {} \n".format(time_limit))
            f.write("TotalSolvingTime {} \n".format(comp_time))
            f.write("TotalSolvingTimeSubproblems {}\n".format(sub_time))
            f.write("NumNode {}\n".format(node_count))
            
            
            #f.write("patient_id,aide_id,visit_time,assigned_aide_day_1,assigned_aide_day_2,assigned_aide_day_3,assigned_aide_day_4,assigned_aide_day_5")
            #f.write('\n')
            #for i in self.list_patient_new_id:
             #   f.write(str(self.instance.list_patient[i].id)+','+str(self.instance.list_patient[i].pre_assigned_aide[0])+
              #          ','+str(self.instance.list_patient[i].pre_assigned_aide[1])+','+
               #        str(self.instance.list_patient[i].pre_assigned_aide[2])+','+str(self.instance.list_patient[i].pre_assigned_aide[3])+','+
                #       str(self.instance.list_patient[i].pre_assigned_aide[4]))
                #f.write('\n')
                
        #"""


        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-AssignmentRelaxation-route_approx.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write('Route_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.route[d]))
                    f.write('\n')

        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-AssignmentRelaxation-schedule_approx.txt"),"w") as f:
            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.schedule)):
                    f.write('Schedule_{}'.format(d+1))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                


        with open(os.path.join(file_path,f"{self.file_name}-NoLearning-AssignmentRelaxation-Route-Schedule_approx.txt"),"w") as f:

            f.write("----------------------------------------------------------------\n")
            f.write(self.instance.instance_name)
            f.write('\n')
            f.write("Method NoLearn and AssignmentRelaxation \n")
            
            for i in self.active_aide:
                f.write('Aide {}'.format(i.id))
                f.write('\n')
                for d in range(len(i.route)):
                    f.write(str(i.route[d]))
                    f.write('\n')
                    f.write(str(i.schedule[d]))
                    f.write('\n')
                    f.write(str(i.visit_count[d]))
                    f.write('\n')
                    f.write('\n')
                f.write('\n')



            
        
        
        
        
