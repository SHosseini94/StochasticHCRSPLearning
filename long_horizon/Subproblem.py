# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:25:37 2024

@author: saeed
"""

"""
This class solves the subproblem, scheduling and routing of patients.
Each subproblem associates with one aide weekly scheduling. 
Thus, it requires to have access to instance, aide_id, and list of patients
assigned to it (which is taken from master problem). 
These are the inputs entered when we create the subproblem object.   

# The jobs this class does are:
# 
# 1. Generate Cut: 
#     It checks the being feasibility of subproblem, which means the 
    subproblem should be feasible for all days individually and 
    also it should satisfy the time consistency and weekly worktime 
    constraints. If it is not the case, we should create the 
    benders cut to add to the master problem, removes this current 
    assignment solution. To check the feasibility of subproblem,
    we can, first, check the feasibility of daily subproblem; 
    so we define a function, check_if_day_feasible 


       a. check if day feasible: create a CP model and add the
       constraints that is, no_overlap between tasks of day, fix
       first and last tasks. check the solve status. If it was infeasible,
       then we would create a cut, otherwise we return the true as output.
       The cut is a kind of strong cut. 


# Asumptions:
# 
#   1. scenarios for each route not for each day/route 
        (it means there isn't any difference for total time for same
         route in different days of a week), since the time window for
        both patients and aides don't change with days.
# 
#   2. Only one fail results in scenario fail. 
# 
# 

    I try to copy the deterministic version to the subproblem, 
    then I will try to add approximation constraints using BlackBox


# It works properly :)
"""


from docplex.cp.model import *
from docplex.cp.expression import *
from docplex.cp.solution import *

import numpy as np
from scipy.stats import poisson
from scipy.stats import norm

import os 
import logging

logger = logging.getLogger(__name__)


class Subproblem():
    
    
    def __init__(self,instance,aide_id,list_Pik,list_visit,list_Pik_frequency,overtime,alpha,beta,TCov,mean_learning,std_learning):
        
        
        self.instance=instance
        self.aide_id=aide_id
        self.list_Pik=list_Pik
        self.list_visit=list_visit
        self.list_Pik_frequency=list_Pik_frequency
        self.alpha=alpha    
        self.beta=beta     
        self.overtime=overtime
        self.TCov=TCov
    
        self.meanLearning=mean_learning
        self.stdLearning=std_learning


        # to keep all the unique assignments during a week


        self.travel_time=[self.instance.distance_matrix[i][j]  for i in range
                          (self.instance.number_node)
                          for j in range(self.instance.number_node)]
        
        self.service_time=[i.visit_duration for i in self.instance.list_patient]

        # include int rounding for service time
        # clamp global visit index to last entry of the learning curve (plateau assumption)
        self.mean_learned_service={d:{i:self.meanLearning[min(self.list_Pik_frequency[d][i]-1, len(self.meanLearning)-1)] for i in self.list_Pik[d] if self.list_Pik[d]} for d in range(self.instance.horizon)}

        self.std_learned_service={d:{i:self.stdLearning[min(self.list_Pik_frequency[d][i]-1, len(self.stdLearning)-1)] for i in self.list_Pik[d] if self.list_Pik[d]} for d in range(self.instance.horizon)}
        
        #print("list Pik is:")
        #print(self.list_Pik)
        #print("mean learned service is:")
        #print(self.mean_learned_service)

        #print("std learned service is:")
        #print(self.std_learned_service)

        #"""
        index=0
        self.assignment_pattern=[]
        self.assignment_pattern_day={} 
        for d in range(self.instance.horizon):
            if self.list_Pik[d] not in self.assignment_pattern and self.list_Pik[d]:
                self.assignment_pattern.append(self.list_Pik[d])
                self.assignment_pattern_day[index]=(d,self.list_Pik[d])
                index+=1
                
        
        self.number_pattern=index
        
        self.number_repeated_pattern=[0 for d in range(self.number_pattern)]
        for d in range(self.instance.horizon):
            for i in range(self.number_pattern):
                if self.list_Pik[d]==self.assignment_pattern[i]:
                    self.number_repeated_pattern[i]+=1

        #self.nb_points=[len(self.assignment_pattern[d])+2 for d in range(self.number_pattern)]
                    
        #"""
        self.nb_points=[len(self.list_Pik[d])+2 for d in range(self.instance.horizon)]

        #print("the number of nodes per day is:")
        #print(self.nb_points)
        

      
        
        self.tasks_per_day=[[] for i in range(self.instance.horizon)]
        self.sequence=[None for i in range(self.instance.horizon)]


        self.work_time_pattern=[0 for i in range(self.number_pattern)]
        self.work_time=[None for i in range(self.instance.horizon)]
        
        self.list_pairs=[]  
        #self.list_cuts={d:[] for d in range(self.instance.horizon)}
        self.list_cuts=[]
        self.dic_cuts={}

        self.time_window_start=self.instance.list_aide[self.aide_id].time_window_start[0]
        self.time_window_end=self.instance.list_aide[self.aide_id].time_window_end[1]
        
        self.solve_daily_sub=0
        self.inf_daily_sub=0
        self.complete_sub=0
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.over_time_status=False

        self.index_patient_removed=[0 for i in range(self.instance.horizon)]

        self.copy_list_Pik={d:self.list_Pik[d].copy() for d in range(self.instance.horizon)}
        



    def solve_complete_problem(self):

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        
        
        file_path=os.path.join(self.BASE_DIR, 'result','learn', 'FinalCallFullModelEVT.txt')
        file=open(file_path,'a')

        infeasible_day=False

        self.sub=CpoModel(name="Cp model of aide_{}".format(self.aide_id+1))

        for d in range(self.instance.horizon):
            if self.list_Pik[d]:

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                        
                        
                tasks=[]
                self.earlist_start[d]=[]
                self.latest_end[d]=[]
                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                    
                    #tasks=interval_var_list(self.nb_points[d],name="tasks_{}_{}".format(self.aide_id,d+1))  # tasks related to this pattern
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work
                    
                    # specify the interval variables' attributes, such as size and start:
                        
                #self.set_tasks_values(d)
                self.similar_set_tasks_values(d)
                
                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq
                
                distance_matrix=self.compute_transition_matrix(d)
                
                self.sub.add(no_overlap(seq,distance_matrix,1))
                self.sub.add(first(seq,self.tasks_per_day[d][0]))
                self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))
                
                

                #arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2)
                arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))
                
                for i in range(self.nb_points[d]):
                    if i==0:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                    elif i==self.nb_points[d]-1:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                    else:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                    
                
                
                    
                stats=arrival_statistics(seq,tasks,d)
                index=0
                
                #"""
                for v in range(1,self.nb_points[d]):


                    if v!=self.nb_points[d]-1:
                        self.sub.add(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1])
                        
                        
                    else:  #only allow to work maximum for overtime longer than their end time window on each day 
                        self.sub.add(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                    
                    index+=2  
        
        self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
               <=self.instance.list_aide[self.aide_id].maximum_work_time)

        self.sub.export_model(file)    
        sol=self.sub.solve(Workers=1,LogVerbosity="Quiet", TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet" 
        if (sol):
            
            logger.info("In last call: The subproblem is solved completely for aide %s", self.aide_id)
            
            index=0
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    self.instance.list_aide[self.aide_id].route.append([])
                    self.instance.list_aide[self.aide_id].schedule.append([])
                    self.instance.list_aide[self.aide_id].visit_count.append([])
                    
                    seq=sol.get_var_solution(self.sequence[d])
                    value=seq.get_value()
                    for i in range(len(value)-1):
                        if i==0:
                            from_location=self.instance.list_aide[self.aide_id].location_start
                            from_time=value[i].get_start()
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()

                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1

                            
                            self.instance.list_aide[self.aide_id].schedule[index].append(from_time)
                            self.instance.list_aide[self.aide_id].route[index].append(from_location)
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)

                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_Pik_frequency[d][to_location-self.instance.number_aide])
                            
                        elif i==len(value)-2:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
                            to_location=self.instance.list_aide[self.aide_id].location_end
                            to_time=self.instance.list_aide[self.aide_id].time_window_end[1]
                            
                            #self.instance.list_aide[self.aide_id].route[d].append(from_location)
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            
                        else:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
            
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()

                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_Pik_frequency[d][to_location-self.instance.number_aide])
                
                    self.instance.list_route.append(self.instance.list_aide[self.aide_id].route[index])
                    self.instance.list_schedule.append(self.instance.list_aide[self.aide_id].schedule[index])
                    self.instance.list_visit_count.append(self.instance.list_aide[self.aide_id].visit_count[index])
                    index+=1
            sol.write(file)
        else:
            logger.error("In last call: The subproblem is not feasible for complete problem for aide %s", self.aide_id)
    

            #print(self.instance.list_route)
                
            #print(self.instance.list_schedule)
                


                

        




    
               
    def generate_cut(self,final_call):

        
    
        # Full model 
        
        #fixed start and end for care giver 
        
        
        # Create a CpoModel 
        #self.sub=CpoModel(name="Cp model of aide_{}".format(self.aide_id+1))
        
        
        if final_call :

            self.solve_complete_problem()
        else:
            
            #status=self.similar_first_subproblem(file)
            #status=self.generate_stronger_cuts_complete_problem(file)
            status=self.generate_stronger_cuts_daily_check()
            
            return self.over_time_status,status,self.solve_daily_sub,self.inf_daily_sub,self.complete_sub,self.dic_cuts




    def generate_stronger_cuts_daily_check(self): #create a list of patients to be removed for each day 

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        blackbox_constraints={}
        all_feasible_day=True

        for p in range(self.number_pattern):

            

            d = self.assignment_pattern_day[p][0]

            value=""
            for b in self.list_Pik[d]:     
                value+=str(b)+"({})".format(self.list_Pik_frequency[d][b])+"-"
            #value+="/"

            

            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                logger.info("checking day %s with pattern %s", d, value)

                self.sub=CpoModel(name="aide_{}_day_{}_pattern_{}".format(self.aide_id+1,d+1,value))

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                self.earlist_start[d]=[]
                self.latest_end[d]=[]

                tasks=[]

                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work

                self.similar_set_tasks_values(d)

                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq

                distance_matrix=self.compute_transition_matrix(d)
                self.sub.add(no_overlap(self.sequence[d],distance_matrix,1))
                self.sub.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                self.sub.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))

                for i in range(self.nb_points[d]):
                    if i==0:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                    elif i==self.nb_points[d]-1:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                    else:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                
                stats=arrival_statistics(seq,tasks,d)

                index=0
            #"""
                blackbox_constraints[d]={}
                for v in range(1,self.nb_points[d]):

                    if v!=self.nb_points[d]-1:

                        blackbox_constraints[d][v]=(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1]) #self.mean_learned_service[d][self.list_Pik[d][v-1]]
                        self.sub.add(blackbox_constraints[d][v])
                    

                    else:  #only allow to work maximum for overtime longer than their end time window on each day
                        blackbox_constraints[d][v]=(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                        self.sub.add(blackbox_constraints[d][v])

                    index+=2

            
            
                sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

                
                if (sol):

                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'FeasibleDailyCheck.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                    #file=open(file_path,'a')
                    self.solve_daily_sub+=1

                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)
                    #self.sub.export_model(file)
                    #sol.write(file)
                    #file.close()


                    logger.info("This pattern is feasible")
                    logger.info("solve time: %s", sol.get_solve_time())
                else:
                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'InfeasibleDailyCheck.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                    #file=open(file_path,'a')

                    #self.sub.export_model(file)    
                    #file.close()

                    all_feasible_day=False
                    self.inf_daily_sub+=1
                    logger.warning("This pattern is infeasible")
                    logger.warning("solve time: %s", sol.get_solve_time())
                    #stronger cuts:
                    remove_index_list=[]

                    if len(self.list_Pik[d])!=1:# when there is only one patient and infeasible on this day, just cut it off
                        
                        for i,patient in enumerate(self.list_Pik[d]):  # only patient


                            
                            self.tasks_per_day[d][i+1].set_absent()
                            self.copy_list_Pik[d].remove(patient)
                            remove_index_list.append((i,patient))

                            

                            self.copy_list_Pik[d].sort()

                            value=""
                            for b in self.copy_list_Pik[d]:     
                                value+=str(b)+"({})".format(self.list_Pik_frequency[d][b])+"-"
                            #value+="/"

                            logger.info("checking sub pattern %s", value)

                            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                                sub_temporary=CpoModel(name="sub_temp_{}_{}_value_{}".format(self.aide_id+1,d+1,value))

                                sub_temporary.add(no_overlap(self.sequence[d],distance_matrix,1))
                                sub_temporary.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                                sub_temporary.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                                sub_temporary.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                                temporary_arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(len(self.copy_list_Pik[d])+1))
                            

                                temporary_stats=temporary_arrival_statistics(self.sequence[d],self.tasks_per_day[d],d)
                                
                                index=0
                    #"""
                                for v in range(len(self.copy_list_Pik[d])+1):

                                    if v!=len(self.copy_list_Pik[d]):

                                        sub_temporary.add(temporary_stats[index]+self.alpha*temporary_stats[index+1]<=self.instance.list_patient[self.copy_list_Pik[d][v]].time_window[1])#self.mean_learned_service[d][self.copy_list_Pik[d][v]]
                                        index+=2

                                    else:  #only allow to work maximum for overtime longer than their end time window on each day
                                        
                                        sub_temporary.add(temporary_stats[index]+self.beta*temporary_stats[index+1]<= self.time_window_end+self.overtime)

                                        index+=2
                                            #self.list_cuts[day].append(i)
                                
                                
                                temporary_sol=sub_temporary.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"
                                
                                if (temporary_sol):
                                    logger.info("by removing patient %s the daily subproblem becomes feasible", patient)
                                    logger.info("solve time is %s", temporary_sol.get_solve_time())
                                    #print("solve time is {}".format(temporary_sol.get_solve_time()))
                                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'FeasibleDailyCheck.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                                    #file=open(file_path,'a')
                                    
                                    
                                    self.tasks_per_day[d][i+1].set_present()
                                    self.copy_list_Pik[d].insert(i,patient)
                                    #sub_temporary.export_model(file)
                                    #temporary_sol.write(file)
                                    #file.close()

                                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)
                            

                                    
                                else:
                                    
                                    #self.tasks_per_day[d][i].set_present()
                                    
                                    self.index_patient_removed[d]+=1

                                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'InfeasibleDailyCheck.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                                    #file=open(file_path,'a')

                                    #sub_temporary.export_model(file)
                                    #file.close()
                                    logger.warning("even by removing patient %s the daily subproblem is still infeasible", patient)
                                    logger.info("solve time is %s", temporary_sol.get_solve_time())

                                    if len(self.copy_list_Pik[d])==1:
                                        #print("only one patient left, stop here")
                                        break
                            else:
                                #print("this pattern is already checked")
                                logger.info("This sub-pattern is already checked")
                                self.tasks_per_day[d][i+1].set_present()
                                self.copy_list_Pik[d].insert(i,patient)



                    self.dic_cuts[d]=[]
                    for item in self.copy_list_Pik[d]:
                        self.list_cuts.append(((item,self.list_Pik_frequency[d][item]),d))   #(patient_id,day)
                        self.dic_cuts[d].append((item,self.list_Pik_frequency[d][item]))
            else:
                logger.info("pattern %s is already checked", value)
                #pass
                    
                            
        #print("index removed for days are {}".format(self.index_patient_removed))
        

        if all_feasible_day:
            # solve complete problem to check for over time 

            value=""
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    for b in self.list_Pik[d]:     
                        value+=str(b)+"({})".format(self.list_Pik_frequency[d][b])+"-"
                    value+="/"

            self.complete_sub=1

            self.sub=CpoModel(name="Complete Cp model of aide_{}_pattern_{}".format(self.aide_id+1,value))

            logger.info("Checking complete subproblem for aide %s with pattern %s", self.aide_id, value)
            self.itv_to_pnt={} 
            self.pnt_id={}
            self.earlist_start={}
            self.latest_end={}

            for d in range(self.instance.horizon):
                
                if self.list_Pik[d]:
                    self.itv_to_pnt[d]={}
                    self.pnt_id[d]={}
                    self.earlist_start[d]=[]
                    self.latest_end[d]=[]

                    tasks=[]

                    for i in range(self.nb_points[d]):
                        if i!=0 and i!=self.nb_points[d]-1:
                            location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                            tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                            self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                            self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                                
                                
                        elif i==0:
                            tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_start)
                            
                        else:
                            tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_end)
                                
                        
                    self.tasks_per_day[d]=tasks
                    work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                    self.work_time[d]=work

                    self.similar_set_tasks_values(d)

                    seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                    self.sequence[d]=seq

                    distance_matrix=self.compute_transition_matrix(d)
                    self.sub.add(no_overlap(seq,distance_matrix,1))
                    self.sub.add(first(seq,self.tasks_per_day[d][0]))
                    self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                    self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                    arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))

                    for i in range(self.nb_points[d]):
                        if i==0:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                            self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                        elif i==self.nb_points[d]-1:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                            self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                        else:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                            self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                    
                    stats=arrival_statistics(seq,tasks,d)

                    index=0
                
                #"""
                    for v in range(1,self.nb_points[d]):

                        if v!=self.nb_points[d]-1:
                            self.sub.add(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1])#-self.mean_learned_service[d][self.list_Pik[d][v-1]]
                        
                        
                        else:  #only allow to work maximum for overtime longer than their end time window on each day 
                            self.sub.add(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                    
                        index+=2 
                
            self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
                         <=self.instance.list_aide[self.aide_id].maximum_work_time)

        
                
            #self.sub.export_model(file) 
                
            sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

            if (sol):
                
                file_name='{}'.format(self.aide_id) +'_' +'FeasibleCompleteSubproblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                file=open(file_path,'a')
                logger.info("The complete subproblem is solved completely for aide %s", self.aide_id)
                self.instance.list_feasible_per_aide[self.aide_id].append(value)
                self.sub.export_model(file)
                sol.write(file)
                file.close()
            else:
                all_feasible_day=False
                logger.warning("The complete subproblem is infeasible for aide %s", self.aide_id)

                file_name='{}'.format(self.aide_id) +'_'+'InfeasibleCompleteSubproblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','learn', file_name)
                file=open(file_path,'a')

                self.sub.export_model(file)
                file.close()

                # simple no-good cut 
                for d in range(self.instance.horizon):
                    if self.list_Pik[d]:
                        self.dic_cuts[d]=[]
                        for item in self.list_Pik[d]:
                            self.list_cuts.append(((item,self.list_Pik_frequency[d][item]),d))   #(patient_id,day)
                            self.dic_cuts[d].append((item,self.list_Pik_frequency[d][item]))



        
        return all_feasible_day








    def similar_set_tasks_values(self,d):
        for i in range(self.nb_points[d]):

            if i==0:    #start base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_size(0)
            elif i==self.nb_points[d]-1:  #end base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_size(0)
            else:   #patient 
                
                index=self.list_Pik[d][i-1]  #patient id 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_patient[index].time_window[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_patient[index].time_window[1]-int(np.round(self.mean_learned_service[d][index])))
                self.tasks_per_day[d][i].set_size(int(np.round(self.mean_learned_service[d][index])))        
                                
    def set_tasks_values(self,d):
        for i in range(self.nb_points[d]):

            if i==0:    #start base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_end_max(self.instance.list_aide[self.aide_id].time_window_start[1])
                self.tasks_per_day[d][i].set_size(0)
            elif i==self.nb_points[d]-1:  #end base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_end_max(self.instance.list_aide[self.aide_id].time_window_end[1]+self.overtime)
                self.tasks_per_day[d][i].set_size(0)
            else:   #patient 
                
                index=self.list_Pik[d][i-1]  #patient id 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_patient[index].time_window[0])
                self.tasks_per_day[d][i].set_end_max(self.instance.list_patient[index].time_window[1])
                self.tasks_per_day[d][i].set_size(int(np.round(self.mean_learned_service[d][index])))
                
    
    
    def get_route_values(self,d):
        
        for i in range(self.nb_points[d]):

            if i==0:
                self.sub.add(self.visit_time[d][i]==self.time_window_start) 
                self.sub.add(self.route[d][i]==self.instance.list_aide[self.aide_id].location_start)
            elif i==self.nb_points[d]-1:
                self.sub.add(self.visit_time[d][i]==self.time_window_end)  
                self.sub.add(self.route[d][i]==self.instance.list_aide[self.aide_id].location_end)
            else:
                #self.sub.add(any([self.visit_time[d][i]==start_of(self.tasks_per_day[d][j]) for j in
                 #                 (1,self.nb_points[d]-1)]))
                
                self.sub.add(any([self.visit_time[d][i]==self.time_window[j] for j in self.assignment_pattern[d]]))
                #"""
                node=0
                #task=[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]
                task=[self.time_window[j] for j in self.assignment_pattern[d]]
                
                for j in task:
                    self.sub.add(if_then(self.visit_time[d][i]==j,self.route[d][i]==
                                     self.assignment_pattern[d][node]+self.instance.number_aide))
                    
                    self.sub.add(if_then(self.route[d][i]==
                                     self.assignment_pattern[d][node]+self.instance.number_aide,self.visit_time[d][i]==j))
                    node+=1
                #""" 
                
    def check_subroute_success(self,d,v):
        
        # d: day v:position in a route 
        
        if v==1:
            total_time=element(self.travel_time,self.route[d][v-1]*self.instance.number_node+self.route[d][v])
        else:
            service_time=element(self.service_time,self.route[d][v-1]-self.instance.number_aide)
            travel_time=element(self.travel_time,self.route[d][v-1]*self.instance.number_node+self.route[d][v])
            total_time=service_time+travel_time
        
        
        
        
        if v==self.nb_points[d]-1:
            
            finish=integer_var(name="finish_work_pattern{}".format(d+1))
            self.sub.add(finish==self.start[d][v-1]+service_time)
            self.sub.add(finish<=self.visit_time[d][v]+self.overtime)
            
        else:
            
            arrival=integer_var(name="arrival_pattern{}_node{}".format(d+1,v+1)) 
            start=integer_var(name="start_pattern{}_node{}".format(d+1,v+1)) 
            
            self.sub.add(arrival==self.start[d][v-1]+total_time)
            self.sub.add(if_then(arrival<self.visit_time[d][v],(start==self.visit_time[d][v])))
        
            self.sub.add(if_then((self.visit_time[d][v]<=arrival),(start==arrival)))
        
            self.sub.add(arrival<=self.visit_time[d][v]+self.L)
        
            self.start[d].append(start)
        
    def compute_transition_matrix(self,k):
        
        transition_matrix=np.zeros((self.nb_points[k],self.nb_points[k]),dtype=int)
        for i in range(self.nb_points[k]):
            if i==0:
                location_from=self.instance.list_aide[self.aide_id].location_start
            elif i==self.nb_points[k]-1:
                location_from=self.instance.list_aide[self.aide_id].location_end
            else:
                location_from=self.instance.list_patient[self.list_Pik[k][i-1]].location
            for j in range(self.nb_points[k]):
                if j==0:
                    location_to=self.instance.list_aide[self.aide_id].location_start
                elif j==self.nb_points[k]-1:
                    location_to=self.instance.list_aide[self.aide_id].location_end
                else:
                    location_to=self.instance.list_patient[self.list_Pik[k][j-1]].location
                        
                transition_matrix[i][j]=self.instance.distance_matrix[location_from][location_to]
                
        
        return transition_matrix
    """
    def approximate_start_time_stats(self,mu1,mu2,theta,sigma1,sigma2):
        
        mean=mu1*norm.cdf((mu1-mu2)/theta)+mu2*norm.cdf((mu2-mu1)/theta)+theta*norm.pdf((mu1-mu2)/theta)
        
        variance=(mu1**2+sigma1**2)*norm.cdf((mu1-mu2)/theta)+(mu2**2+sigma2**2)*norm.cdf((mu2-mu1)/theta)+(mu1+mu2)*theta*norm.pdf((mu1-mu2)/theta)-mean**2
        std=np.sqrt(variance)
        
        return mean, std
    """
    
    #for only one pattern 
    
    def approximate_start_time_stats(self,mu1,mu2,theta,sigma1,sigma2):
        
        mean=mu1*norm.cdf((mu1-mu2)/theta)+mu2*norm.cdf((mu2-mu1)/theta)+theta*norm.pdf((mu1-mu2)/theta)
        
        variance=(mu1**2+sigma1**2)*norm.cdf((mu1-mu2)/theta)+(mu2**2+sigma2**2)*norm.cdf((mu2-mu1)/theta)+(mu1+mu2)*theta*norm.pdf((mu1-mu2)/theta)-mean**2
        
        
        if variance<0:
            variance=0
        
        #std=np.sqrt(variance)
        std=variance
        
        
        return mean, std
    
    
    #for only one pattern 
    
    def arrival_approximation(self,sequence,tasks,d):  #all_visit_time
        
        route=[]
        visit_time=[]
        index_task=[]
        #print("day is {}".format(d))
        #print("earlist start is {}".format(self.earlist_start[d]))
        for i,itv in enumerate(sequence):
            order=self.itv_to_pnt[d][itv]
            route.append(order)
            index=self.pnt_id[d][order]
            index_task.append(index)
            if index==0:
                visit_time.append(tasks[0].start)
            else:
                visit_time.append(self.earlist_start[d][index])
        
        #index_task=self.pnt_id[d][route[position]]
        #print(index_task)

        #print("route is {}".format(route))
        #print("visit time is {}".format(visit_time))

        
        num_node=len(route)-1
        
        mean_arrival=np.empty(num_node,dtype=np.float64)  #to keep mean of arrival time for each node
        std_arrival=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of arrival time for each node 
        
        mean_start=np.empty(num_node,dtype=np.float64)  #to keep mean of start time for each node
        std_start=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of start time for each node 

        mean_finish=np.empty(num_node,dtype=np.float64)  #to keep mean of end time for each node
        std_finish=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of end time for each node
        
        all_stat=[]
        #print(route)
        #print(visit_time)
        
        
        for i in range(1,num_node+1):
            
            if i == 1:
                
                
                from_i=route[i-1]
                to_j=route[i]
                
                mean_arrival[i-1]=visit_time[i-1]+self.instance.distance_matrix[from_i][to_j]
                
                std_arrival[i-1]=self.instance.distance_matrix[from_i][to_j]*self.TCov
                
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                sigma1=std_arrival[i-1]
                sigma2=0
                
                theta=sigma1+0.000001
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
            
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                visit_duration_mean_j=self.mean_learned_service[d][to_j-self.instance.number_aide]
                visit_duration_std_j=self.std_learned_service[d][to_j-self.instance.number_aide]

                mean_finish[i-1]=mean_start[i-1]+visit_duration_mean_j
                std_finish[i-1]=np.sqrt(std_start[i-1]+visit_duration_std_j**2)
                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
                
                
            
            elif i==num_node:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                # finish work time 
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i
                
                variance=std_start[i-2]+visit_duration_std_i**2
                
                std_arrival[i-1]=np.sqrt(variance)

                mean_finish[i-1]=mean_arrival[i-1]
                std_finish[i-1]=std_arrival[i-1]
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
                
                
            else:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                #print("visit duration mean of {} is {}".format(from_i,visit_duration_mean_i))
                #print("visit duration std of {} is {}".format(from_i,visit_duration_std_i))

                #print("travel time from {} to {} is {}".format(from_i,to_j,self.instance.distance_matrix[from_i][to_j]))
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i+self.instance.distance_matrix[from_i][to_j]
                
                variance=std_start[i-2]+visit_duration_std_i**2+(self.instance.distance_matrix[from_i][to_j]*self.TCov)**2
                std_arrival[i-1]=np.sqrt(variance)
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                
                sigma1=std_arrival[i-1]
                sigma2=0
                theta=sigma1+0.000001
                
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                mean_finish[i-1]=mean_start[i-1]+self.mean_learned_service[d][to_j-self.instance.number_aide]
                std_finish[i-1]=np.sqrt(std_start[i-1]+self.std_learned_service[d][to_j-self.instance.number_aide]**2)
                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
            
            #all_stat.append(int(self.pnt_id[d][route[i]]))
            all_stat.append((index_task[i],mean_finish[i-1],std_finish[i-1]))
            
            
                
                
                
                #mean_arrival, std_arrival
        #print(mean_arrival)
        
        #print(std_arrival)
        
    
        #print("arrival mean for position {} is {} and arrival standard deviation is {}".format(position,mean_arrival[position-1],std_arrival[position-1]))
        all_stat=sorted(all_stat,key=lambda x: x[0])

        #print(all_stat)

        all_stat_2=[]
        for i in range(len(all_stat)):
            all_stat_2.append(all_stat[i][1])
            all_stat_2.append(all_stat[i][2])

        

        
        
        #print(all_stat_2)
        return all_stat_2
                
    
    
    
    def arrival_approximation_2(self,sequence,d,patient): #visit_time
        
        
        route=[]
        visit_time=[]
        
        for i,itv in enumerate(sequence):
            order=self.itv_to_pnt[d][itv]
            route.append(order)
            visit_time.append(itv.get_start())
        
        print(route)
        print(visit_time)
        
        index_patient=route.index(patient+self.instance.number_aide)
        print(index_patient)
        
        
        
        if index_patient!=1:
            
            prev=[route[i] for i in range(1,index_patient+1)] 
            num_node=len(prev)
            
            print(prev)
            
            mean_arrival=np.empty(num_node,dtype=np.float64)
            std_arrival=np.empty(num_node,dtype=np.float64)
            
            mean_start=np.empty(num_node,dtype=np.float64)
            std_start=np.empty(num_node,dtype=np.float64)
            
            for i in range(index_patient):
                
                from_i=route[i]
                to_j=route[i+1]
                
                if i==0:
                    mean_arrival[i]=visit_time[i]+self.instance.distance_matrix[from_i][to_j]
                    std_arrival[i]=self.instance.distance_matrix[from_i][to_j]*self.TCov
                
                    mu1=mean_arrival[i]
                    #time_window[self.time_pnt[to_j-self.instance.number_aide]]
                    mu2=visit_time[i+1]  #sequence[i+1].start # visit_time[i+1] 
                
                    sigma1=std_arrival[i]
                    sigma2=0
                
                
                    theta=sigma1
                
                    mean_start[i],std_start[i]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)
                
                else:
                    
                    visit_duration_mean_i=self.instance.list_patient[from_i-self.instance.number_aide].visit_duration
                    visit_duration_std_i=self.SCov*visit_duration_mean_i
                
                    mean_arrival[i]=mean_start[i-1]+visit_duration_mean_i+self.instance.distance_matrix[from_i][to_j]
                
                    variance=std_start[i-1]**2+visit_duration_std_i**2+(self.instance.distance_matrix[from_i][to_j]*self.TCov)**2
                    std_arrival[i]=np.sqrt(variance)
                
                
                    mu1=mean_arrival[i]
                    mu2=visit_time[i+1] #sequence[i+1].start 
                
                
                    sigma1=std_arrival[i]
                    sigma2=0
                    theta=sigma1
                
                    mean_start[i],std_start[i]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)
                
            
            mean_arrival_patient=mean_arrival[-1]
            std_arrival_patient=std_arrival[-1]
                
            
        
        else:
        
            from_i=route[0]
            to_j=route[1]
            mean_arrival_patient=visit_time[0]+self.instance.distance_matrix[from_i][to_j]
            std_arrival_patient=self.instance.distance_matrix[from_i][to_j]*self.TCov
      
                
        print(visit_time[0])        
            
        print(mean_arrival_patient)
        
        print(std_arrival_patient)
        #print(node_id)
        #print("arrival mean for position {} is {} and arrival standard deviation is {}".format(position,mean_arrival[position-1],std_arrival[position-1]))
        return mean_arrival_patient,std_arrival_patient
                
        
        
        
    
class No_Learn_Subproblem():
    def __init__(self,instance,aide_id,list_Pik,list_visit,overtime,alpha,beta,TCov,mean_noLearn,std_noLearn):
        
        
        self.instance=instance
        self.aide_id=aide_id
        self.list_Pik=list_Pik
        self.alpha=alpha    
        self.beta=beta     
        self.overtime=overtime
        self.TCov=TCov
        self.list_visit=list_visit
        # Prior visits for this aide specifically: {patient_id: count_before_this_run}
        _all_prior = getattr(self.instance, 'prior_visit_counts', {})
        self._prior_visits = {pat_id: aide_dict.get(self.aide_id, 0)
                              for pat_id, aide_dict in _all_prior.items()}

        # include int rounding for service time
        self.mean_learned_service={d:{i:mean_noLearn[0] for i in self.list_Pik[d] if self.list_Pik[d]} for d in range(self.instance.horizon)}

        self.std_learned_service={d:{i:std_noLearn[0] for i in self.list_Pik[d] if self.list_Pik[d]} for d in range(self.instance.horizon)}
        
        #print("list Pik is:")
        #print(self.list_Pik)
        #print("mean learned service is:")
        #print(self.mean_learned_service)

        #print("std learned service is:")
        #print(self.std_learned_service)

        #"""
        index=0
        self.assignment_pattern=[]
        self.assignment_pattern_day={} 
        for d in range(self.instance.horizon):
            if self.list_Pik[d] not in self.assignment_pattern and self.list_Pik[d]:
                self.assignment_pattern.append(self.list_Pik[d])
                self.assignment_pattern_day[index]=(d,self.list_Pik[d])
                index+=1
                
        
        self.number_pattern=index
        
        self.number_repeated_pattern=[0 for d in range(self.number_pattern)]
        for d in range(self.instance.horizon):
            for i in range(self.number_pattern):
                if self.list_Pik[d]==self.assignment_pattern[i]:
                    self.number_repeated_pattern[i]+=1

        #self.nb_points=[len(self.assignment_pattern[d])+2 for d in range(self.number_pattern)]
                    
        #"""
        self.nb_points=[len(self.list_Pik[d])+2 for d in range(self.instance.horizon)]

        #print("the number of nodes per day is:")
        #print(self.nb_points)
        

      
        
        self.tasks_per_day=[[] for i in range(self.instance.horizon)]
        self.sequence=[None for i in range(self.instance.horizon)]


        self.work_time_pattern=[0 for i in range(self.number_pattern)]
        self.work_time=[None for i in range(self.instance.horizon)]
        
        self.list_pairs=[]  
        #self.list_cuts={d:[] for d in range(self.instance.horizon)}
        self.list_cuts=[]
        self.dic_cuts={}

        self.time_window_start=self.instance.list_aide[self.aide_id].time_window_start[0]
        self.time_window_end=self.instance.list_aide[self.aide_id].time_window_end[1]
        
        self.solve_daily_sub=0
        self.inf_daily_sub=0
        self.complete_sub=0
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.over_time_status=False

        self.index_patient_removed=[0 for i in range(self.instance.horizon)]

        self.copy_list_Pik={d:self.list_Pik[d].copy() for d in range(self.instance.horizon)}
    
    
    

    def solve_complete_problem(self):

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        
        
        file_path=os.path.join(self.BASE_DIR, 'result','no-learn','NoLearnFinalCallFullModelEVT.txt')
        file=open(file_path,'a')

        infeasible_day=False

        self.sub=CpoModel(name="Cp model of aide_{}_".format(self.aide_id+1))

        for d in range(self.instance.horizon):
            if self.list_Pik[d]:

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                        
                        
                tasks=[]
                self.earlist_start[d]=[]
                self.latest_end[d]=[]
                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                    
                    #tasks=interval_var_list(self.nb_points[d],name="tasks_{}_{}".format(self.aide_id,d+1))  # tasks related to this pattern
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work
                    
                    # specify the interval variables' attributes, such as size and start:
                        
                self.similar_set_tasks_values(d)
                
                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq
                
                distance_matrix=self.compute_transition_matrix(d)
                
                self.sub.add(no_overlap(seq,distance_matrix,1))
                self.sub.add(first(seq,self.tasks_per_day[d][0]))
                self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))
                
                

                #arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2)
                arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))
                
                for i in range(self.nb_points[d]):
                    if i==0:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                    elif i==self.nb_points[d]-1:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                    else:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                    
                
                
                    
                stats=arrival_statistics(seq,tasks,d)
                index=0
                
                #"""
                for v in range(1,self.nb_points[d]):


                    if v!=self.nb_points[d]-1:
                        self.sub.add(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1]) #self.mean_learned_service[d][self.list_Pik[d][v-1]]
                        
                        
                    else:  #only allow to work maximum for overtime longer than their end time window on each day 
                        self.sub.add(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                    
                    index+=2  
        
        self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
               <=self.instance.list_aide[self.aide_id].maximum_work_time)

        self.sub.export_model(file)    
        sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet" 
        if (sol):
            logger.info("In final call:The complete subproblem is solved completely for aide %s", self.aide_id)
            index=0
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    self.instance.list_aide[self.aide_id].route.append([])
                    self.instance.list_aide[self.aide_id].schedule.append([])
                    self.instance.list_aide[self.aide_id].visit_count.append([])
                    
                    
                    seq=sol.get_var_solution(self.sequence[d])
                    value=seq.get_value()
                    for i in range(len(value)-1):
                        if i==0:
                            from_location=self.instance.list_aide[self.aide_id].location_start
                            from_time=value[i].get_start()
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()

                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1
                            
                            self.instance.list_aide[self.aide_id].schedule[index].append(from_time)
                            self.instance.list_aide[self.aide_id].route[index].append(from_location)
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_visit[d][to_location-self.instance.number_aide] + self._prior_visits.get(to_location-self.instance.number_aide, 0))
                            
                        elif i==len(value)-2:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
                            to_location=self.instance.list_aide[self.aide_id].location_end
                            to_time=self.instance.list_aide[self.aide_id].time_window_end[1]
                            
                            #self.instance.list_aide[self.aide_id].route[d].append(from_location)
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            
                        else:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
            
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()


                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_visit[d][to_location-self.instance.number_aide] + self._prior_visits.get(to_location-self.instance.number_aide, 0))
                            
                
                    self.instance.list_route.append(self.instance.list_aide[self.aide_id].route[index])
                    self.instance.list_schedule.append(self.instance.list_aide[self.aide_id].schedule[index])
                    self.instance.list_visit_count.append(self.instance.list_aide[self.aide_id].visit_count[index])
                    
                    index+=1
            sol.write(file)
        else:
            logger.error("In last call: The subproblem is not feasible for complete problem for aide %s", self.aide_id)
            

            #print(self.instance.list_route)
                
            #print(self.instance.list_schedule)



        
    def generate_cut(self,final_call):

        
    
        # Full model 
        
        #fixed start and end for care giver 
        
        
        # Create a CpoModel 
        #self.sub=CpoModel(name="Cp model of aide_{}".format(self.aide_id+1))
        
        
        
        if final_call:

            self.solve_complete_problem()
        else:

        
            status=self.generate_stronger_cuts_daily_check()
            
            return self.over_time_status,status,self.solve_daily_sub,self.inf_daily_sub,self.complete_sub,self.dic_cuts
    

    def generate_stronger_cuts_daily_check(self): #create a list of patients to be removed for each day 

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        blackbox_constraints={}
        all_feasible_day=True

        for p in range(self.number_pattern):

            

            d = self.assignment_pattern_day[p][0]

            value=""
            for b in self.list_Pik[d]:     
                value+=str(b)+"-"
            #value+="/"

            

            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                #print("checking day {} with pattern {}".format(d,value))
                logger.info("Checking day %s with pattern %s", d, value)

                self.sub=CpoModel(name="aide_{}_day_{}_pattern_{}".format(self.aide_id+1,d+1,value))

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                self.earlist_start[d]=[]
                self.latest_end[d]=[]

                tasks=[]

                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work

                self.similar_set_tasks_values(d)

                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq

                distance_matrix=self.compute_transition_matrix(d)
                self.sub.add(no_overlap(self.sequence[d],distance_matrix,1))
                self.sub.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                self.sub.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))

                for i in range(self.nb_points[d]):
                    if i==0:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                    elif i==self.nb_points[d]-1:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                        self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                    else:
                        self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                
                stats=arrival_statistics(seq,tasks,d)

                index=0
            #"""
                blackbox_constraints[d]={}
                for v in range(1,self.nb_points[d]):

                    if v!=self.nb_points[d]-1:

                        blackbox_constraints[d][v]=(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1]) #self.mean_learned_service[d][self.list_Pik[d][v-1]]
                        self.sub.add(blackbox_constraints[d][v])
                    

                    else:  #only allow to work maximum for overtime longer than their end time window on each day
                        blackbox_constraints[d][v]=(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                        self.sub.add(blackbox_constraints[d][v])

                    index+=2

            
            
                sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

                if (sol):
                    self.solve_daily_sub+=1

                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)

                    #file_name='{}'.format(self.aide_id) +'_'+value+ '_' +'NoLearnFeasibleDailySubproblem.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                    #file=open(file_path,'a')
                    #self.sub.export_model(file)
                    #sol.write(file)
                    #file.close()



                    #print("This pattern is feasible")
                    logger.info("This pattern is feasible")
                    #print("solve time is {}".format(sol.get_solve_time()))
                    logger.info("Solve time is %s", sol.get_solve_time())
                else:
                    all_feasible_day=False
                    self.inf_daily_sub+=1
                    #print("This pattern is infeasible")
                    logger.warning("This pattern is infeasible")
                    #print("solve time is {}".format(sol.get_solve_time()))
                    logger.warning("Solve time is %s", sol.get_solve_time())
                    #stronger cuts:
                    remove_index_list=[]

                    #file_name='{}'.format(self.aide_id) +'_'+value+ '_' +'NoLearnInfeasibleDailySubproblem.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                    #file=open(file_path,'a')
                    #self.sub.export_model(file)
                    #file.close()


                    if len(self.list_Pik[d])!=1:# when there is only one patient and infeasible on this day, just cut it off
                        
                        for i,patient in enumerate(self.list_Pik[d]):  # only patient


                            
                            self.tasks_per_day[d][i+1].set_absent()
                            self.copy_list_Pik[d].remove(patient)
                            remove_index_list.append((i,patient))

                            

                            self.copy_list_Pik[d].sort()

                            value=""
                            for b in self.copy_list_Pik[d]:     
                                value+=str(b)+"-"
                            #value+="/"

                            #print("checking sub pattern {}".format(value))
                            logger.info("Checking sub-pattern %s", value)
                            

                            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                                sub_temporary=CpoModel(name="sub_temp_{}_{}_pattern_{}".format(self.aide_id+1,d+1,value))

                                sub_temporary.add(no_overlap(self.sequence[d],distance_matrix,1))
                                sub_temporary.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                                sub_temporary.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                                sub_temporary.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                                temporary_arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(len(self.copy_list_Pik[d])+1))
                            

                                temporary_stats=temporary_arrival_statistics(self.sequence[d],self.tasks_per_day[d],d)
                                
                                index=0
                    #"""
                                for v in range(len(self.copy_list_Pik[d])+1):

                                    if v!=len(self.copy_list_Pik[d]):

                                        sub_temporary.add(temporary_stats[index]+self.alpha*temporary_stats[index+1]<=self.instance.list_patient[self.copy_list_Pik[d][v]].time_window[1]) #self.mean_learned_service[d][self.copy_list_Pik[d][v]]
                                        index+=2

                                    else:  #only allow to work maximum for overtime longer than their end time window on each day
                                        
                                        sub_temporary.add(temporary_stats[index]+self.beta*temporary_stats[index+1]<= self.time_window_end+self.overtime)

                                        index+=2
                                            #self.list_cuts[day].append(i)
                                
                                
                                temporary_sol=sub_temporary.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

                                if (temporary_sol):
                                    #print("by removing patient {} the daily subproblem becomes feasible".format(patient))
                                    logger.info("By removing patient %s the daily subproblem becomes feasible", patient)
                                    #print("solve time is {}".format(temporary_sol.get_solve_time()))
                                    logger.info("Solve time is %s", temporary_sol.get_solve_time())
                                    
                                    self.tasks_per_day[d][i+1].set_present()
                                    self.copy_list_Pik[d].insert(i,patient)


                                    #file_name='{}'.format(self.aide_id) +'_' + value +'_'+'NoLearnFeasibleDailySubproblem.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                                    #file=open(file_path,'a')
                                    #sub_temporary.export_model(file)
                                    #temporary_sol.write(file)
                                    #file.close()

                                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)
                            

                                    
                                else:
                                    logger.warning("This sub-pattern is infeasible")
                                    logger.warning("Solve time is %s", temporary_sol.get_solve_time())

                                    #file_name='{}'.format(self.aide_id) +'_' + value +'_'+'NoLearnInfeasibleDailySubproblem.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                                    #file=open(file_path,'a')
                                    #sub_temporary.export_model(file)
                                    #file.close()
                                    
                                    #self.tasks_per_day[d][i].set_present()
                                    self.index_patient_removed[d]+=1

                                    if len(self.copy_list_Pik[d])==1:
                                        #print("only one patient left, stop here")
                                        break
                            else:
                                #print("this pattern is already checked")
                                logger.info("This sub-pattern %s is already checked", value)
                                self.tasks_per_day[d][i+1].set_present()
                                self.copy_list_Pik[d].insert(i,patient)



                    self.dic_cuts[d]=[]
                    for item in self.copy_list_Pik[d]:
                        self.list_cuts.append((item,d))   #(patient_id,day)
                        self.dic_cuts[d].append(item)
            else:
                #print("pattern {} is already checked".format(value))
                logger.info("Pattern %s is already checked for aide %s", value, self.aide_id)
                pass
                    
                            
        #print("index removed for days are {}".format(self.index_patient_removed))
        

        if all_feasible_day:
            # solve complete problem to check for over time 

            self.complete_sub=1
            value=""
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    for b in self.list_Pik[d]:     
                        value+=str(b)+"-"
                    value+="/"

            self.sub=CpoModel(name="Complete Cp model of aide_{}_value_{}".format(self.aide_id+1,value))

            self.itv_to_pnt={} 
            self.pnt_id={}
            self.earlist_start={}
            self.latest_end={}

            

            for d in range(self.instance.horizon):
                
                if self.list_Pik[d]:
                    self.itv_to_pnt[d]={}
                    self.pnt_id[d]={}
                    self.earlist_start[d]=[]
                    self.latest_end[d]=[]

                    tasks=[]

                    for i in range(self.nb_points[d]):
                        if i!=0 and i!=self.nb_points[d]-1:
                            location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                            tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                            self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                            self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                                
                                
                        elif i==0:
                            tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_start)
                            
                        else:
                            tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_end)
                                
                        
                    self.tasks_per_day[d]=tasks
                    work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                    self.work_time[d]=work

                    self.similar_set_tasks_values(d)

                    seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                    self.sequence[d]=seq

                    distance_matrix=self.compute_transition_matrix(d)
                    self.sub.add(no_overlap(seq,distance_matrix,1))
                    self.sub.add(first(seq,self.tasks_per_day[d][0]))
                    self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                    self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                    arrival_statistics=CpoBlackboxFunction(impl=self.arrival_approximation,dimension=2*(self.nb_points[d]-1))

                    for i in range(self.nb_points[d]):
                        if i==0:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_start
                            self.pnt_id[d][self.instance.list_aide[self.aide_id].location_start]=i
                        elif i==self.nb_points[d]-1:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_aide[self.aide_id].location_end
                            self.pnt_id[d][self.instance.list_aide[self.aide_id].location_end]=i
                        else:
                            self.itv_to_pnt[d][self.tasks_per_day[d][i]]=self.instance.list_patient[self.list_Pik[d][i-1]].location
                            self.pnt_id[d][self.instance.list_patient[self.list_Pik[d][i-1]].location]=i
                    
                    stats=arrival_statistics(seq,tasks,d)

                    index=0
                
                #"""
                    for v in range(1,self.nb_points[d]):

                        if v!=self.nb_points[d]-1:
                            self.sub.add(stats[index]+self.alpha*stats[index+1]<= self.latest_end[d][v-1]) #self.mean_learned_service[d][self.list_Pik[d][v-1]]
                        
                        
                        else:  #only allow to work maximum for overtime longer than their end time window on each day 
                            self.sub.add(stats[index]+self.beta*stats[index+1]<= self.time_window_end+self.overtime)
                    
                        index+=2 
                
            self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
                         <=self.instance.list_aide[self.aide_id].maximum_work_time)

            
                
            #self.sub.export_model(file) 
                
            sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

            if (sol):
                #print("The complete subproblem is solved completely for aide {}".format(self.aide_id))
                logger.info("The complete subproblem is solved completely for aide %s", self.aide_id)

                file_name='{}'.format(self.aide_id) +'_' +'NoLearnFeasibleCompleteSubproblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                file=open(file_path,'a')
                self.sub.export_model(file)
                sol.write(file)
                file.close()
                self.instance.list_feasible_per_aide[self.aide_id].append(value)
                
            else:
                all_feasible_day=False
                logger.warning("The complete subproblem is not feasible for aide %s", self.aide_id)

                file_name='{}'.format(self.aide_id) +'_' +'NoLearnInfeasibleCompleteSubproblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn', file_name)
                file=open(file_path,'a')
                self.sub.export_model(file)
                file.close()

                # simple no-good cut 
                for d in range(self.instance.horizon):
                    if self.list_Pik[d]:
                        self.dic_cuts[d]=[]
                        for item in self.list_Pik[d]:
                            self.list_cuts.append((item,d))   #(patient_id,day)
                            self.dic_cuts[d].append(item)



        
        return all_feasible_day
    


    def similar_set_tasks_values(self,d):
        for i in range(self.nb_points[d]):

            if i==0:    #start base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_size(0)
            elif i==self.nb_points[d]-1:  #end base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_size(0)
            else:   #patient 
                
                index=self.list_Pik[d][i-1]  #patient id 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_patient[index].time_window[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_patient[index].time_window[1]-int(np.round(self.mean_learned_service[d][index])))
                self.tasks_per_day[d][i].set_size(int(np.round(self.mean_learned_service[d][index])))
    


    def compute_transition_matrix(self,k):
        
        transition_matrix=np.zeros((self.nb_points[k],self.nb_points[k]),dtype=int)
        for i in range(self.nb_points[k]):
            if i==0:
                location_from=self.instance.list_aide[self.aide_id].location_start
            elif i==self.nb_points[k]-1:
                location_from=self.instance.list_aide[self.aide_id].location_end
            else:
                location_from=self.instance.list_patient[self.list_Pik[k][i-1]].location
            for j in range(self.nb_points[k]):
                if j==0:
                    location_to=self.instance.list_aide[self.aide_id].location_start
                elif j==self.nb_points[k]-1:
                    location_to=self.instance.list_aide[self.aide_id].location_end
                else:
                    location_to=self.instance.list_patient[self.list_Pik[k][j-1]].location
                        
                transition_matrix[i][j]=self.instance.distance_matrix[location_from][location_to]
                
        
        return transition_matrix
    



    def approximate_start_time_stats(self,mu1,mu2,theta,sigma1,sigma2):
        
        mean=mu1*norm.cdf((mu1-mu2)/theta)+mu2*norm.cdf((mu2-mu1)/theta)+theta*norm.pdf((mu1-mu2)/theta)
        
        variance=(mu1**2+sigma1**2)*norm.cdf((mu1-mu2)/theta)+(mu2**2+sigma2**2)*norm.cdf((mu2-mu1)/theta)+(mu1+mu2)*theta*norm.pdf((mu1-mu2)/theta)-mean**2
        
        
        if variance<0:
            variance=0
        
        #std=np.sqrt(variance)
        std=variance
        
        
        return mean, std
    
    
    #for only one pattern 
    
    def arrival_approximation(self,sequence,tasks,d):  #all_visit_time
        
        route=[]
        visit_time=[]
        index_task=[]
        #print("day is {}".format(d))
        #print("earlist start is {}".format(self.earlist_start[d]))
        for i,itv in enumerate(sequence):
            order=self.itv_to_pnt[d][itv]
            route.append(order)
            index=self.pnt_id[d][order]
            index_task.append(index)
            if index==0:
                visit_time.append(tasks[0].start)
            else:
                visit_time.append(self.earlist_start[d][index])
        
        #index_task=self.pnt_id[d][route[position]]
        #print(index_task)

        #print("route is {}".format(route))
        #print("visit time is {}".format(visit_time))
        
        num_node=len(route)-1
        
        mean_arrival=np.empty(num_node,dtype=np.float64)  #to keep mean of arrival time for each node
        std_arrival=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of arrival time for each node 
        
        mean_start=np.empty(num_node,dtype=np.float64)  #to keep mean of start time for each node
        std_start=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of start time for each node 

        mean_finish=np.empty(num_node,dtype=np.float64)  #to keep mean of finish time for each node
        std_finish=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of finish time for each node

        all_stat=[]
        #print(route)
        #print(visit_time)
        
        
        for i in range(1,num_node+1):
            
            if i == 1:
                
                
                from_i=route[i-1]
                to_j=route[i]
                
                mean_arrival[i-1]=visit_time[i-1]+self.instance.distance_matrix[from_i][to_j]
                
                std_arrival[i-1]=self.instance.distance_matrix[from_i][to_j]*self.TCov
                
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                sigma1=std_arrival[i-1]
                sigma2=0
                
                theta=sigma1+0.000001
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
            
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                mean_finish[i-1]=mean_start[i-1]+self.mean_learned_service[d][to_j-self.instance.number_aide]
                
                std_finish[i-1]=np.sqrt(std_start[i-1]+self.std_learned_service[d][to_j-self.instance.number_aide]**2)

                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
                
                
            
            elif i==num_node:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                # finish work time 
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i
                
                variance=std_start[i-2]+visit_duration_std_i**2
                
                std_arrival[i-1]=np.sqrt(variance)

                mean_finish[i-1]=mean_arrival[i-1]
                std_finish[i-1]=std_arrival[i-1]


                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
                
                
            else:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                #print("visit duration mean of {} is {}".format(from_i,visit_duration_mean_i))
                #print("visit duration std of {} is {}".format(from_i,visit_duration_std_i))

                #print("travel time from {} to {} is {}".format(from_i,to_j,self.instance.distance_matrix[from_i][to_j]))
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i+self.instance.distance_matrix[from_i][to_j]
                
                variance=std_start[i-2]+visit_duration_std_i**2+(self.instance.distance_matrix[from_i][to_j]*self.TCov)**2
                std_arrival[i-1]=np.sqrt(variance)
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                
                sigma1=std_arrival[i-1]
                sigma2=0
                theta=sigma1+0.000001
                
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                mean_finish[i-1]=mean_start[i-1]+self.mean_learned_service[d][to_j-self.instance.number_aide]
                std_finish[i-1]=np.sqrt(std_start[i-1]+self.std_learned_service[d][to_j-self.instance.number_aide]**2)
                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
            
            #all_stat.append(int(self.pnt_id[d][route[i]]))
            all_stat.append((index_task[i],mean_finish[i-1],std_finish[i-1]))
            
            
                
                
                
                #mean_arrival, std_arrival
        #print(mean_arrival)
        
        #print(std_arrival)
        
    
        #print("arrival mean for position {} is {} and arrival standard deviation is {}".format(position,mean_arrival[position-1],std_arrival[position-1]))
        all_stat=sorted(all_stat,key=lambda x: x[0])

        #print(all_stat)

        all_stat_2=[]
        for i in range(len(all_stat)):
            all_stat_2.append(all_stat[i][1])
            all_stat_2.append(all_stat[i][2])

        

        
        
        #print(all_stat_2)
        return all_stat_2
    



class Deterministic_No_Learn_Subproblem():
    def __init__(self,instance,aide_id,list_Pik,list_visit,overtime,TCov,mean_noLearn):
        
        
        self.instance=instance
        self.aide_id=aide_id
        self.list_Pik=list_Pik    
        self.overtime=overtime
        self.TCov=TCov
        self.list_visit=list_visit
        # Prior visits for this aide specifically: {patient_id: count_before_this_run}
        _all_prior = getattr(self.instance, 'prior_visit_counts', {})
        self._prior_visits = {pat_id: aide_dict.get(self.aide_id, 0)
                              for pat_id, aide_dict in _all_prior.items()}

        # include int rounding for service time
        self.mean_learned_service={d:{i:mean_noLearn[0] for i in self.list_Pik[d] if self.list_Pik[d]} for d in range(self.instance.horizon)}


        #print("list Pik is:")
        #print(self.list_Pik)
        #print("mean learned service is:")
        #print(self.mean_learned_service)

        #print("std learned service is:")
        #print(self.std_learned_service)

        #"""
        index=0
        self.assignment_pattern=[]
        self.assignment_pattern_day={} 
        for d in range(self.instance.horizon):
            if self.list_Pik[d] not in self.assignment_pattern and self.list_Pik[d]:
                self.assignment_pattern.append(self.list_Pik[d])
                self.assignment_pattern_day[index]=(d,self.list_Pik[d])
                index+=1
                
        
        self.number_pattern=index
        
        self.number_repeated_pattern=[0 for d in range(self.number_pattern)]
        for d in range(self.instance.horizon):
            for i in range(self.number_pattern):
                if self.list_Pik[d]==self.assignment_pattern[i]:
                    self.number_repeated_pattern[i]+=1

        #self.nb_points=[len(self.assignment_pattern[d])+2 for d in range(self.number_pattern)]
                    
        #"""
        self.nb_points=[len(self.list_Pik[d])+2 for d in range(self.instance.horizon)]

        #print("the number of nodes per day is:")
        #print(self.nb_points)
        

      
        
        self.tasks_per_day=[[] for i in range(self.instance.horizon)]
        self.sequence=[None for i in range(self.instance.horizon)]


        self.work_time_pattern=[0 for i in range(self.number_pattern)]
        self.work_time=[None for i in range(self.instance.horizon)]
        
        self.list_pairs=[]  
        #self.list_cuts={d:[] for d in range(self.instance.horizon)}
        self.list_cuts=[]
        self.dic_cuts={}

        self.time_window_start=self.instance.list_aide[self.aide_id].time_window_start[0]
        self.time_window_end=self.instance.list_aide[self.aide_id].time_window_end[1]
        
        self.solve_daily_sub=0
        self.inf_daily_sub=0
        self.complete_sub=0
        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.over_time_status=False

        self.index_patient_removed=[0 for i in range(self.instance.horizon)]

        self.copy_list_Pik={d:self.list_Pik[d].copy() for d in range(self.instance.horizon)}
    
    
    

    def solve_complete_problem(self):

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        
        
        file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', 'Deterministic-NoLearnFinalCallFullModelEVT.txt')
        file=open(file_path,'a')

        infeasible_day=False

        self.sub=CpoModel(name="Cp model of aide_{}".format(self.aide_id+1))

        for d in range(self.instance.horizon):
            if self.list_Pik[d]:

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                        
                        
                tasks=[]
                self.earlist_start[d]=[]
                self.latest_end[d]=[]
                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                    
                    #tasks=interval_var_list(self.nb_points[d],name="tasks_{}_{}".format(self.aide_id,d+1))  # tasks related to this pattern
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work
                    
                    # specify the interval variables' attributes, such as size and start:
                        
                self.similar_set_tasks_values(d)
                
                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq
                
                distance_matrix=self.compute_transition_matrix(d)
                
                self.sub.add(no_overlap(seq,distance_matrix,1))
                self.sub.add(first(seq,self.tasks_per_day[d][0]))
                self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))
                
    
                
                
                    
                
        
        self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
               <=self.instance.list_aide[self.aide_id].maximum_work_time)

        self.sub.export_model(file)    
        sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet" 
        if (sol):
            
            logger.info("In last call: The subproblem is feasible for complete problem for aide %s", self.aide_id)
            index=0
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    self.instance.list_aide[self.aide_id].route.append([])
                    self.instance.list_aide[self.aide_id].schedule.append([])
                    self.instance.list_aide[self.aide_id].visit_count.append([])
                    
                    
                    seq=sol.get_var_solution(self.sequence[d])
                    value=seq.get_value()
                    for i in range(len(value)-1):
                        if i==0:
                            from_location=self.instance.list_aide[self.aide_id].location_start
                            from_time=value[i].get_start()
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()

                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1
                            
                            self.instance.list_aide[self.aide_id].schedule[index].append(from_time)
                            self.instance.list_aide[self.aide_id].route[index].append(from_location)
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_visit[d][to_location-self.instance.number_aide] + self._prior_visits.get(to_location-self.instance.number_aide, 0))
                            
                        elif i==len(value)-2:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
                            to_location=self.instance.list_aide[self.aide_id].location_end
                            to_time=self.instance.list_aide[self.aide_id].time_window_end[1]
                            
                            #self.instance.list_aide[self.aide_id].route[d].append(from_location)
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            
                        else:
                            from_name=value[i].get_name()
                            m,n,location,b=from_name.split('_',3)
                            from_location=int(location)
            
                            
                            to_name=value[i+1].get_name()
                            m,n,location,b=to_name.split('_',3)
                            to_location=int(location)
                            to_time=value[i+1].get_start()


                            if self.aide_id in self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]+=1
                            else:
                                self.instance.list_patient[to_location-self.instance.number_aide].aide_assigned[self.aide_id]=1
                            
                            self.instance.list_aide[self.aide_id].route[index].append(to_location)
                            self.instance.list_aide[self.aide_id].schedule[index].append(to_time)
                            self.instance.list_aide[self.aide_id].visit_count[index].append(self.list_visit[d][to_location-self.instance.number_aide] + self._prior_visits.get(to_location-self.instance.number_aide, 0))
                            
                
                    self.instance.list_route.append(self.instance.list_aide[self.aide_id].route[index])
                    self.instance.list_schedule.append(self.instance.list_aide[self.aide_id].schedule[index])
                    self.instance.list_visit_count.append(self.instance.list_aide[self.aide_id].visit_count[index])
                    
                    index+=1
            sol.write(file)
        else:
            logger.error("In last call: The subproblem is not feasible for complete problem for aide %s", self.aide_id)
    
        file.close()
            #print(self.instance.list_route)
                
            #print(self.instance.list_schedule)



        
    def generate_cut(self,final_call):

        
    
        # Full model 
        
        #fixed start and end for care giver 
        
        
        # Create a CpoModel 
        #self.sub=CpoModel(name="Cp model of aide_{}".format(self.aide_id+1))
        
        
        
        if final_call:

            self.solve_complete_problem()
        else:

            
            status=self.generate_stronger_cuts_daily_check()
            
            return self.over_time_status,status,self.solve_daily_sub,self.inf_daily_sub,self.complete_sub,self.dic_cuts
    

    def generate_stronger_cuts_daily_check(self): #create a list of patients to be removed for each day 

        self.itv_to_pnt={} 
        self.pnt_id={}
        self.earlist_start={}
        self.latest_end={}
        blackbox_constraints={}
        all_feasible_day=True

        for p in range(self.number_pattern):

            

            d = self.assignment_pattern_day[p][0]

            value=""
            for b in self.list_Pik[d]:     
                value+=str(b)+"-"
            #value+="/"

            

            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                #print("checking day {} with pattern {}".format(d,value))
                logger.info("Checking day %s with pattern %s", d, value)

                self.sub=CpoModel(name="aide_{}_day_{}_pattern_{}".format(self.aide_id+1,d+1,value))

                self.itv_to_pnt[d]={}
                self.pnt_id[d]={}
                self.earlist_start[d]=[]
                self.latest_end[d]=[]

                tasks=[]

                for i in range(self.nb_points[d]):
                    if i!=0 and i!=self.nb_points[d]-1:
                        location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                        tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                        self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                        self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                            
                            
                    elif i==0:
                        tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_start)
                        
                    else:
                        tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                        self.earlist_start[d].append(self.time_window_end)
                            
                    
                self.tasks_per_day[d]=tasks
                work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                self.work_time[d]=work

                self.similar_set_tasks_values(d)

                seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                self.sequence[d]=seq

                distance_matrix=self.compute_transition_matrix(d)
                self.sub.add(no_overlap(self.sequence[d],distance_matrix,1))
                self.sub.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                self.sub.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                self.sub.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                

            
            
                sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

                if (sol):
                    self.solve_daily_sub+=1

                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)

                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'DeterministicFeasibleDailyCheck.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                    #file=open(file_path,'a')

                    #self.sub.export_model(file)

                    #sol.write(file)
                    #file.close()


                    #print("This pattern is feasible")
                    logger.info("This pattern is feasible")
                    logger.info("Solve time is %s", sol.get_solve_time())
                    #print("solve time is {}".format(sol.get_solve_time()))
                else:
                    all_feasible_day=False
                    self.inf_daily_sub+=1
                    #print("This pattern is infeasible")
                    logger.warning("This pattern is infeasible")
                    #print("solve time is {}".format(sol.get_solve_time()))
                    logger.info("Solve time is %s", sol.get_solve_time())
                    
                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'DeterministicInfeasibleDailyCheck.txt'
                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                    #file=open(file_path,'a')
                    #self.sub.export_model(file)
                    #file.close()
                    remove_index_list=[]

                    if len(self.list_Pik[d])!=1:# when there is only one patient and infeasible on this day, just cut it off
                        
                        for i,patient in enumerate(self.list_Pik[d]):  # only patient


                            
                            self.tasks_per_day[d][i+1].set_absent()
                            self.copy_list_Pik[d].remove(patient)
                            remove_index_list.append((i,patient))

                            

                            self.copy_list_Pik[d].sort()

                            value=""
                            for b in self.copy_list_Pik[d]:     
                                value+=str(b)+"-"
                            #value+="/"

                            #print("checking sub pattern {}".format(value))
                            logger.info("Checking sub pattern %s", value)

                            if value not in self.instance.list_feasible_day_per_aide[self.aide_id]:

                                sub_temporary=CpoModel(name="sub_temp_{}_{}_pattern_{}".format(self.aide_id+1,d+1,value))

                                sub_temporary.add(no_overlap(self.sequence[d],distance_matrix,1))
                                sub_temporary.add(first(self.sequence[d],self.tasks_per_day[d][0]))
                                sub_temporary.add(last(self.sequence[d],self.tasks_per_day[d][self.nb_points[d]-1]))
                                sub_temporary.add(span(self.work_time[d],[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                                
                                
                                temporary_sol=sub_temporary.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

                                if (temporary_sol):
                                    #print("by removing patient {} the daily subproblem becomes feasible".format(patient))
                                    logger.info("By removing patient %s the daily subproblem becomes feasible", patient)
                                    #print("solve time is {}".format(temporary_sol.get_solve_time()))
                                    logger.info("Solve time is %s", temporary_sol.get_solve_time())
                                    
                                    self.tasks_per_day[d][i+1].set_present()
                                    self.copy_list_Pik[d].insert(i,patient)

                                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'DeterministicFeasibleDailyCheck.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                                    #file=open(file_path,'a')
                                    #sub_temporary.export_model(file)
                                    #temporary_sol.write(file)
                                    #file.close()

                                    self.instance.list_feasible_day_per_aide[self.aide_id].append(value)
                            

                                    
                                else:

                                    #file_name='{}'.format(self.aide_id) +'_'+ value +'_'+'DeterministicInfeasibleDailyCheck.txt'
                                    #file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                                    #file=open(file_path,'a')
                                    #sub_temporary.export_model(file)
                                    #file.close()
                                    #self.tasks_per_day[d][i].set_present()
                                    self.index_patient_removed[d]+=1

                                    logger.warning("By removing patient %s the daily subproblem is still infeasible", patient)
                                    logger.info("Solve time is %s", temporary_sol.get_solve_time())

                                    if len(self.copy_list_Pik[d])==1:
                                        #print("only one patient left, stop here")
                                        break
                                
                            else:
                                #print("this pattern is already checked")
                                logger.info("This pattern is already checked")
                                self.tasks_per_day[d][i+1].set_present()
                                self.copy_list_Pik[d].insert(i,patient)



                    self.dic_cuts[d]=[]
                    for item in self.copy_list_Pik[d]:
                        self.list_cuts.append((item,d))   #(patient_id,day)
                        self.dic_cuts[d].append(item)
            else:
                #print("pattern {} is already checked".format(value))
                logger.info("Pattern %s is already checked", value)
                pass
                    
                            
        #print("index removed for days are {}".format(self.index_patient_removed))
        

        if all_feasible_day:
            # solve complete problem to check for over time 


            value=""
            for d in range(self.instance.horizon):
                if self.list_Pik[d]:
                    for b in self.list_Pik[d]:     
                        value+=str(b)+"-"
                    value+="/"

            self.complete_sub=1

            self.sub=CpoModel(name="Complete Cp model of aide_{}_pattern_{}".format(self.aide_id+1,value))

            self.itv_to_pnt={} 
            self.pnt_id={}
            self.earlist_start={}
            self.latest_end={}

            for d in range(self.instance.horizon):
                
                if self.list_Pik[d]:
                    self.itv_to_pnt[d]={}
                    self.pnt_id[d]={}
                    self.earlist_start[d]=[]
                    self.latest_end[d]=[]

                    tasks=[]

                    for i in range(self.nb_points[d]):
                        if i!=0 and i!=self.nb_points[d]-1:
                            location=self.instance.list_patient[self.list_Pik[d][i-1]].location
                            tasks.append(interval_var(name="task_pnt_{}_{}".format(location,d)))
                            self.earlist_start[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[0])
                            self.latest_end[d].append(self.instance.list_patient[self.list_Pik[d][i-1]].time_window[1])
                                
                                
                        elif i==0:
                            tasks.append(interval_var(name="task_start_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_start)
                            
                        else:
                            tasks.append(interval_var(name="task_end_depot_{}".format(d)))
                            self.earlist_start[d].append(self.time_window_end)
                                
                        
                    self.tasks_per_day[d]=tasks
                    work=interval_var(name="work_time_{}_{}".format(self.aide_id+1,d+1))   # total work time of this pattern
                    self.work_time[d]=work

                    self.similar_set_tasks_values(d)

                    seq=sequence_var(self.tasks_per_day[d],types=[i for i in range(self.nb_points[d])],name="sequence_{}_{}".format(self.aide_id+1,d+1))
                    self.sequence[d]=seq

                    distance_matrix=self.compute_transition_matrix(d)
                    self.sub.add(no_overlap(seq,distance_matrix,1))
                    self.sub.add(first(seq,self.tasks_per_day[d][0]))
                    self.sub.add(last(seq,self.tasks_per_day[d][self.nb_points[d]-1]))
                    self.sub.add(span(work,[self.tasks_per_day[d][j] for j in range(1,self.nb_points[d]-1)]))

                    
                
            self.sub.add(sum([length_of(self.work_time[d]) for d in range(self.instance.horizon) if self.list_Pik[d]])
                         <=self.instance.list_aide[self.aide_id].maximum_work_time)

            
                
            #self.sub.export_model(file) 
                
            sol=self.sub.solve(Workers=1,LogVerbosity="Quiet",TimeLimit=10)  #Workers=4 TimeLimit=10 LogVerbosity="Quiet"

            if (sol):
                #print("The complete subproblem is solved completely for aide {}".format(self.aide_id))
                logger.info("The complete subproblem is solved completely for aide %s", self.aide_id)
                self.instance.list_feasible_per_aide[self.aide_id].append(value)

                file_name='{}'.format(self.aide_id) +'_'+'DeterministicFeasibleCompleteProblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                file=open(file_path,'a')
                self.sub.export_model(file)
                sol.write(file)
                file.close()
            else:
                # simple no-good cut 
                all_feasible_day=False
                logger.warning("The complete subproblem is infeasible for aide %s", self.aide_id)

                file_name='{}'.format(self.aide_id) +'_'+'DeterministicInfeasibleCompleteProblem.txt'
                file_path=os.path.join(self.BASE_DIR, 'result','no-learn','deterministic', file_name)
                file=open(file_path,'a')
                self.sub.export_model(file)
                file.close()

                for d in range(self.instance.horizon):
                    if self.list_Pik[d]:
                        self.dic_cuts[d]=[]
                        for item in self.list_Pik[d]:
                            self.list_cuts.append((item,d))   #(patient_id,day)
                            self.dic_cuts[d].append(item)



        
        return all_feasible_day
    


    def similar_set_tasks_values(self,d):
        for i in range(self.nb_points[d]):

            if i==0:    #start base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_start[0])
                self.tasks_per_day[d][i].set_size(0)
            elif i==self.nb_points[d]-1:  #end base 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_aide[self.aide_id].time_window_end[1])
                self.tasks_per_day[d][i].set_size(0)
            else:   #patient 
                
                index=self.list_Pik[d][i-1]  #patient id 
                self.tasks_per_day[d][i].set_start_min(self.instance.list_patient[index].time_window[0])
                self.tasks_per_day[d][i].set_start_max(self.instance.list_patient[index].time_window[1]-int(np.round(self.mean_learned_service[d][index])))
                self.tasks_per_day[d][i].set_size(int(np.round(self.mean_learned_service[d][index])))
    


    def compute_transition_matrix(self,k):
        
        transition_matrix=np.zeros((self.nb_points[k],self.nb_points[k]),dtype=int)
        for i in range(self.nb_points[k]):
            if i==0:
                location_from=self.instance.list_aide[self.aide_id].location_start
            elif i==self.nb_points[k]-1:
                location_from=self.instance.list_aide[self.aide_id].location_end
            else:
                location_from=self.instance.list_patient[self.list_Pik[k][i-1]].location
            for j in range(self.nb_points[k]):
                if j==0:
                    location_to=self.instance.list_aide[self.aide_id].location_start
                elif j==self.nb_points[k]-1:
                    location_to=self.instance.list_aide[self.aide_id].location_end
                else:
                    location_to=self.instance.list_patient[self.list_Pik[k][j-1]].location
                        
                transition_matrix[i][j]=self.instance.distance_matrix[location_from][location_to]
                
        
        return transition_matrix
    



    def approximate_start_time_stats(self,mu1,mu2,theta,sigma1,sigma2):
        
        mean=mu1*norm.cdf((mu1-mu2)/theta)+mu2*norm.cdf((mu2-mu1)/theta)+theta*norm.pdf((mu1-mu2)/theta)
        
        variance=(mu1**2+sigma1**2)*norm.cdf((mu1-mu2)/theta)+(mu2**2+sigma2**2)*norm.cdf((mu2-mu1)/theta)+(mu1+mu2)*theta*norm.pdf((mu1-mu2)/theta)-mean**2
        
        
        if variance<0:
            variance=0
        
        #std=np.sqrt(variance)
        std=variance
        
        
        return mean, std
    
    
    #for only one pattern 
    
    def arrival_approximation(self,sequence,tasks,d):  #all_visit_time
        
        route=[]
        visit_time=[]
        index_task=[]
        #print("day is {}".format(d))
        #print("earlist start is {}".format(self.earlist_start[d]))
        for i,itv in enumerate(sequence):
            order=self.itv_to_pnt[d][itv]
            route.append(order)
            index=self.pnt_id[d][order]
            index_task.append(index)
            if index==0:
                visit_time.append(tasks[0].start)
            else:
                visit_time.append(self.earlist_start[d][index])
        
        #index_task=self.pnt_id[d][route[position]]
        #print(index_task)

        #print("route is {}".format(route))
        #print("visit time is {}".format(visit_time))
        
        num_node=len(route)-1
        
        mean_arrival=np.empty(num_node,dtype=np.float64)  #to keep mean of arrival time for each node
        std_arrival=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of arrival time for each node 
        
        mean_start=np.empty(num_node,dtype=np.float64)  #to keep mean of start time for each node
        std_start=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of start time for each node 

        mean_finish=np.empty(num_node,dtype=np.float64)  #to keep mean of finish time for each node
        std_finish=np.empty(num_node,dtype=np.float64)   # to keep standard deviation of finish time for each node

        all_stat=[]
        #print(route)
        #print(visit_time)
        
        
        for i in range(1,num_node+1):
            
            if i == 1:
                
                
                from_i=route[i-1]
                to_j=route[i]
                
                mean_arrival[i-1]=visit_time[i-1]+self.instance.distance_matrix[from_i][to_j]
                
                std_arrival[i-1]=self.instance.distance_matrix[from_i][to_j]*self.TCov
                
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                sigma1=std_arrival[i-1]
                sigma2=0
                
                theta=sigma1+0.000001
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
            
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                mean_finish[i-1]=mean_start[i-1]+self.mean_learned_service[d][to_j-self.instance.number_aide]
                
                std_finish[i-1]=np.sqrt(std_start[i-1]+self.std_learned_service[d][to_j-self.instance.number_aide]**2)

                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
                
                
            
            elif i==num_node:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                # finish work time 
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i
                
                variance=std_start[i-2]+visit_duration_std_i**2
                
                std_arrival[i-1]=np.sqrt(variance)

                mean_finish[i-1]=mean_arrival[i-1]
                std_finish[i-1]=std_arrival[i-1]


                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                
                
                
            else:
                
                from_i=route[i-1]
                to_j= route[i]
                
                #service time at node i 
                
                visit_duration_mean_i=self.mean_learned_service[d][from_i-self.instance.number_aide]
                visit_duration_std_i=self.std_learned_service[d][from_i-self.instance.number_aide]
                
                #print("visit duration mean of {} is {}".format(from_i,visit_duration_mean_i))
                #print("visit duration std of {} is {}".format(from_i,visit_duration_std_i))

                #print("travel time from {} to {} is {}".format(from_i,to_j,self.instance.distance_matrix[from_i][to_j]))
                mean_arrival[i-1]=mean_start[i-2]+visit_duration_mean_i+self.instance.distance_matrix[from_i][to_j]
                
                variance=std_start[i-2]+visit_duration_std_i**2+(self.instance.distance_matrix[from_i][to_j]*self.TCov)**2
                std_arrival[i-1]=np.sqrt(variance)
                
                #print("mean_arrival of {} is {}".format(to_j,mean_arrival[i-1]))
                #print("st_arrival of {} is {}".format(to_j,std_arrival[i-1]))
                mu1=mean_arrival[i-1]
                mu2=visit_time[i]
                
                
                sigma1=std_arrival[i-1]
                sigma2=0
                theta=sigma1+0.000001
                
                mean_start[i-1],std_start[i-1]=self.approximate_start_time_stats(mu1,mu2,theta,sigma1,sigma2)

                mean_finish[i-1]=mean_start[i-1]+self.mean_learned_service[d][to_j-self.instance.number_aide]
                std_finish[i-1]=np.sqrt(std_start[i-1]+self.std_learned_service[d][to_j-self.instance.number_aide]**2)
                
                #print("mean_start of {} is {}".format(to_j,mean_start[i-1]))
                #print("std_start of {} is {}".format(to_j,std_start[i-1]))
            
            #all_stat.append(int(self.pnt_id[d][route[i]]))
            all_stat.append((index_task[i],mean_finish[i-1],std_finish[i-1]))
            
            
                
                
                
                #mean_arrival, std_arrival
        #print(mean_arrival)
        
        #print(std_arrival)
        
    
        #print("arrival mean for position {} is {} and arrival standard deviation is {}".format(position,mean_arrival[position-1],std_arrival[position-1]))
        all_stat=sorted(all_stat,key=lambda x: x[0])

        #print(all_stat)

        all_stat_2=[]
        for i in range(len(all_stat)):
            all_stat_2.append(all_stat[i][1])
            all_stat_2.append(all_stat[i][2])

        

        
        
        #print(all_stat_2)
        return all_stat_2