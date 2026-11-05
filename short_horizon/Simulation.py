# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:56:18 2024

@author: saeed
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class simulation():
    
    def __init__(self):
        
        #print("simulation created")
        self.service_level=[]
        self.aide_service_level=[]
        self.record_lateness_overtime={'client':[],'caregiver':[]} # to track the lateness and overtime distributions for instances 
    
        
    def total_delay(self,instance,scenario,overtime,alpha,beta):
        
        nb_scenario=scenario.number_scenario
        
        total_lateness=np.zeros(nb_scenario,dtype=float)  #total delay for all routes of the instance 
        total_overtime=np.zeros(nb_scenario,dtype=float)   #total overtime for all routes of the instance 
        
        run_lateness=np.zeros(nb_scenario,dtype=int)   #if there is any lateness for patients in a run 
        run_overtime=np.zeros(nb_scenario,dtype=int)  #if there is overtime for any of aides in a run 

        total_traveltime=np.zeros(nb_scenario,dtype=float)  #total travel time for all routes of the instance 
        total_waitingtime=np.zeros(nb_scenario,dtype=float)  #total waiting time for all routes of the instance

        

        for i in range(len(instance.list_route)):

            #print("route number {}".format(i))
            
            route=instance.list_route[i]
            origin_start=instance.list_aide[route[0]].time_window_start[0]
            start_visit_time=[instance.list_patient[j-instance.number_aide].time_window[0] for j in route[1:-1]]
            origin_end=instance.list_aide[route[-1]-instance.number_aide-instance.number_patient].time_window_end[1]  #adding the aide's end time at the end of visit_time list
            end_visit_time=[instance.list_patient[j-instance.number_aide].time_window[1] for j in route[1:-1]]

            visit_count=instance.list_visit_count[i]
            #print("route is {}".format(route))
            #print("visit count is {}".format(visit_count))
            #logger.info(f"Processing route {i} with visit count {visit_count}")
            
            total_delay,overal_state,aide_overtime,aide_state,travel_time,waiting_time=self.route_delay(scenario,route,visit_count,origin_start,start_visit_time,end_visit_time,origin_end,instance,overtime)
        
            
            total_lateness+=total_delay
            total_overtime+=aide_overtime

            total_traveltime+=travel_time
            total_waitingtime+=waiting_time
            
            
            for s in range(nb_scenario):
                if run_lateness[s]==0 and overal_state[s]==1:
                    run_lateness[s]=1
                if run_overtime[s]==0 and aide_state[s]==1:
                    run_overtime[s]=1 
        
        
        average_total_lateness=total_lateness.sum()/nb_scenario
        average_total_overtime=total_overtime.sum()/nb_scenario
        
        average_run_lateness=run_lateness.sum()/nb_scenario
        average_run_overtime=run_overtime.sum()/nb_scenario

        average_total_traveltime=total_traveltime.sum()/nb_scenario
        average_total_waitingtime=total_waitingtime.sum()/nb_scenario
        
        max_total_lateness=total_lateness.max()
        
        
        patient_service_record={}
        visit_service_record={}
        aide_service_record={}

        for i,k in self.service_level:
            if i not in patient_service_record:
                patient_service_record[i]=[k]
                visit_service_record[i]=[k]
            else:
                patient_service_record[i].append(k)
                visit_service_record[i].append(k)
                
        for i in patient_service_record.keys():
            patient_service_record[i]=min(patient_service_record[i])


        for i,k in self.aide_service_level:
            if i not in aide_service_record:
                aide_service_record[i]=[k]
            else:
                aide_service_record[i].append(k)
        aide_min_service=min(v for values in aide_service_record.values() for v in values)
        aide_mean_service=sum(v for values in aide_service_record.values() for v in values)/sum(len(values) for values in aide_service_record.values())

        patient_service_record=sorted(patient_service_record.items())
        visit_service_record=sorted(visit_service_record.items())
        aide_service_record=sorted(aide_service_record.items())
        
        value=[]
        visit_value=[]

        patient_below_alpha=[]
        visit_below_alpha=[]
        aide_below_beta=[]
        
        for i in patient_service_record:
            value.append(i[1])
            if round(i[1],2)<alpha:
                patient_below_alpha.append(1)

        for i in visit_service_record:
            for j in range(len(i[1])):
                visit_value.append(i[1][j])

                if round(i[1][j],2)<alpha:
                    visit_below_alpha.append(1)

        
        for i in aide_service_record:
            for j in range(len(i[1])): 
                if round(i[1][j],2)<beta:
                    aide_below_beta.append(1)
        
        nb_patient=len(patient_service_record)
        nb_visit=0
        for i in visit_service_record:
            nb_visit+=len(i[1])

        patient_below_alpha=(nb_patient,sum(patient_below_alpha))
        visit_below_alpha=(nb_visit,sum(visit_below_alpha))
        aide_below_beta=(len(instance.list_route),sum(aide_below_beta))
        
        patient_mean_service=sum(value)/len(value)
        patient_min_service=min(value)

        visit_mean_service=sum(visit_value)/len(visit_value)
        visit_min_service=min(visit_value)

        
        
        return average_total_traveltime,average_total_waitingtime,average_total_lateness,average_total_overtime,average_run_lateness,average_run_overtime,max_total_lateness,patient_service_record,patient_mean_service,patient_min_service,visit_service_record,visit_mean_service,visit_min_service,patient_below_alpha,visit_below_alpha,aide_service_record,aide_min_service,aide_mean_service,aide_below_beta,self.record_lateness_overtime
    
            
            
        
        
    def route_delay(self,scenario,route,visit_count,origin_start,start_visit_time,end_visit_time,origin_end,instance,overtime):
        
        num_node=len(route)-1
        num_scenario=scenario.number_scenario
        arrival=np.empty((num_scenario,num_node),dtype=float)  #to keep mean of arrival time for each node
        start=np.empty((num_scenario,num_node),dtype=float)  #to keep mean of start time for each node
        finish=np.empty((num_scenario,num_node),dtype=float)  #to keep mean of finish time for each node

        state=np.zeros((num_scenario,num_node),dtype=int)   # whether each patient is visited with delay for each scenario or not 
        overal_state=np.zeros(num_scenario,dtype=int)    #if we have any delay for each patient for each scenario it gets 1
        
        total_delay=np.zeros(num_scenario,dtype=float)  #the sum of delay for patients in each scenario 
        
        aide_overtime=np.zeros(num_scenario,dtype=float)  # the amount of overtime for aide in each scenario 
        aide_state=np.zeros(num_scenario,dtype=int)   #whether the aide works overtime in each scenario or not 

        total_traveltime=np.zeros(num_scenario,dtype=float)  #the sum of travel time between locations in each scenario 
        total_waitingtime=np.zeros(num_scenario,dtype=float)  #the sum of waiting time for patients in each scenario 
    
    
        
        for i in range(1,num_node+1):
            #print("node is {}".format(i))
        
            if i==1:
            
                #print(" the node : {}".format(i))
            
                from_i=route[i-1]
                to_j=route[i]
                #print("from_i {} to_j {}".format(from_i,to_j))


                for s in range(num_scenario):
                
                    #print("scenario {} for node {}".format(s,i))
                
                    arrival[s][i-1]=origin_start+scenario.travel_time[s][from_i*instance.number_node+to_j]
                    #print("arrival: {}".format(arrival[s][i-1]))
                    start[s][i-1]=max(start_visit_time[i-1],arrival[s][i-1])
                    #print("start: {}".format(start[s][i-1]))
                    
                    # Only fetch service time for patient nodes; aides/depot have zero service.
                    if to_j < instance.number_aide or to_j >= instance.number_node - instance.number_aide:
                        service_duration = 0
                    else:
                        # visit_count is 1-indexed for service_time dictionary
                        visit_day = visit_count[i-1]
                        try:
                            service_duration = scenario.service_time[to_j][visit_day][s]
                        except KeyError as e:
                            print(f"KeyError: node={to_j}, visit_day={visit_day}, scenario={s}, visit_count[{i-1}]={visit_count[i-1]}")
                            print(f"Available keys for node {to_j}: {list(scenario.service_time.get(to_j, {}).keys())}")
                            logger.error(f"KeyError encountered: {e}")
                            logger.info(f"node={to_j}, visit_day={visit_day}, scenario={s}, visit_count[{i-1}]={visit_count[i-1]}")
                            logger.info(f"Available keys for node {to_j}: {list(scenario.service_time.get(to_j, {}).keys())}")
                            raise
                    
                    finish[s][i-1]=start[s][i-1]+service_duration
                    #print("finish: {}".format(finish[s][i-1]))


                    total_traveltime[s]+=scenario.travel_time[s][from_i*instance.number_node+to_j]
                    #print("travel time added: {}".format(scenario.travel_time[s][from_i*instance.number_node+to_j]))
                    total_waitingtime[s]+=max(0,start_visit_time[i-1]-arrival[s][i-1])
                    #print("waiting time added: {}".format(max(0,start_visit_time[i-1]-arrival[s][i-1])))
                    
                
                    #print("arrival: {}".format(arrival[s][i-1]))
                
                    #print("start: {}".format(start[s][i-1]))
                
                    if finish[s][i-1]>end_visit_time[i-1]:
                        state[s][i-1]=1
                        total_delay[s]+=finish[s][i-1]-end_visit_time[i-1]
                        overal_state[s]=1
                        self.record_lateness_overtime['client'].append(finish[s][i-1]-end_visit_time[i-1])
                        
                    #print("state is {}".format(state[s][i-1]))
                    #print("total delay is {}".format(total_delay[s]))
                    #print("overal state is {}".format(overal_state[s]))
        
            elif i==num_node:
            
                #print(" the node : {}".format(i))
                from_i=route[i-1]
                to_j= route[i]
                #print("from_i {} to_j {}".format(from_i,to_j))
                
                
                for s in range(num_scenario):
                
                    #print("scenario {} for node {}".format(s,i))
                
                    arrival[s][i-1]=finish[s][i-2]
                
                    #print("arrival: {}".format(arrival[s][i-1]))
                
                    if arrival[s][i-1]>origin_end+overtime:
                    
                        state[s][i-1]=1
                        aide_overtime[s]=arrival[s][i-1]-origin_end-overtime
                        aide_state[s]=1
                        self.record_lateness_overtime['caregiver'].append(arrival[s][i-1]-origin_end-overtime)
                
                    #print("state is {}".format(state[s][i-1]))
                    #print("aide overtime is {}".format(aide_overtime[s]))
                    #print("aide state is {}".format(aide_state[s]))
                    
                    
            
            else:
                #print(" the node : {}".format(i))
                
                from_i=route[i-1]
                to_j= route[i]
                #print("from_i {} to_j {}".format(from_i,to_j))
                                
                for s in range(num_scenario):
                
                    #print("scenario {} for node {}".format(s,i))
                
                    arrival[s][i-1]=finish[s][i-2]+scenario.travel_time[s][from_i*instance.number_node+to_j]
                    #print("arrival: {}".format(arrival[s][i-1]))
                    start[s][i-1]=max(start_visit_time[i-1],arrival[s][i-1])
                    #print("start: {}".format(start[s][i-1]))
                    
                    # Only fetch service time for patient nodes; aides/depot have zero service.
                    if to_j < instance.number_aide or to_j >= instance.number_node - instance.number_aide:
                        service_duration = 0
                    else:
                        # visit_count is 1-indexed for service_time dictionary
                        visit_day = visit_count[i-1]
                        try:
                            service_duration = scenario.service_time[to_j][visit_day][s]
                        except KeyError as e:
                            print(f"KeyError: node={to_j}, visit_day={visit_day}, scenario={s}, visit_count[{i-1}]={visit_count[i-1]}")
                            print(f"Available keys for node {to_j}: {list(scenario.service_time.get(to_j, {}).keys())}")
                            raise
                    
                    finish[s][i-1]=start[s][i-1]+service_duration
                    #print("finish: {}".format(finish[s][i-1]))

                    total_traveltime[s]+=scenario.travel_time[s][from_i*instance.number_node+to_j]
                    total_waitingtime[s]+=max(0,start_visit_time[i-1]-arrival[s][i-1])
                    #print("travel time added: {}".format(scenario.travel_time[s][from_i*instance.number_node+to_j]))
                    #print("waiting time added: {}".format(max(0,start_visit_time[i-1]-arrival[s][i-1])))
                
                
                    #print("arrival: {}".format(arrival[s][i-1]))
                    #print("start: {}".format(start[s][i-1]))
                
                    if finish[s][i-1]>end_visit_time[i-1]:
                    
                        state[s][i-1]=1
                        total_delay[s]+=finish[s][i-1]-end_visit_time[i-1]
                        overal_state[s]=1
                        self.record_lateness_overtime['client'].append(finish[s][i-1]-end_visit_time[i-1])
                    #print("total delay is {}".format(total_delay[s]))
                    #print("overal state is {}".format(overal_state[s]))
                    #print("state is {}".format(state[s][i-1]))
                   
                
         #print("state is {}".format(state[s][i-1]))        
                
        service_level=[(route[i]-instance.number_aide,1-state[:,i-1].sum()/num_scenario) for i in range(1,num_node)]
        #print("service level for route {} is {}".format(route,service_level))
        for i in range(len(service_level)):
            self.service_level.append(service_level[i])

        self.aide_service_level.append((route[0],1-aide_state.sum()/num_scenario))
    
   

        return total_delay,overal_state,aide_overtime,aide_state,total_traveltime,total_waitingtime      #np.sum(state,axis=0)/num_scenario,state
                
                    
           
                    

    
    
    

