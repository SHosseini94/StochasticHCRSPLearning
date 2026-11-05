# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:15:42 2024

@author: saeed
"""

"""
 This class is used to read information from text file 
 containing data related to an instance. 
 Also, it write this data into instance object created by Instance class. 
"""

import numpy as np



class Instance_Reader():
    
    def __init__(self):
        
        #print("Instance Reader created")
        pass
    
    def print_instance(self,instance):
        with open("instance_EVT.txt","a") as f:
            f.write(instance.instance_name)
            f.write('\n')
            f.write("*****Patients_Info*****\n")
            f.write("----------------------------------------------------------------\n")
            f.write("patient_id,number_visit,visit_duration,code_value,time_window")
            f.write('\n')
            for i in range(instance.number_patient):
                if instance.list_patient[i].availability==2:
                    f.write(str(instance.list_patient[i].id)+','+str(instance.list_patient[i].number_visit)+','+
                            str(instance.list_patient[i].visit_duration)+','+ str(instance.list_patient[i].code_value)+','+
                            str(instance.list_patient[i].time_window)+','+ '\n')
            f.write("*****Aides_Info*****\n")
            f.write("----------------------------------------------------------------\n")
            f.write("aide_id,time_window_start,time_window_end\n")
            for i in range(instance.number_aide):
                if instance.list_aide[i].availability==2:
                    f.write(str(instance.list_aide[i].id)+','+str(instance.list_aide[i].time_window_start)+','+
                           str(instance.list_aide[i].time_window_end)+'\n')
                    
            
        
    def read_instance(self,file,instance):
        #print("the instance is being read")
        for i in file:
            l=np.array(i.split(" "))
            if "nPatients" in l:
                number_node=int(l[1])
            elif "nNurses" in l:
                number_aide=int(l[1])
                number_patient=number_node-2*number_aide
            elif "nDays" in l:
                number_day=int(l[1])
                instance.initilize(number_patient,number_aide,number_day)
            elif "nurseAvail\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_aide):
                    instance.list_aide[j].availability=int(d[j])
            elif "patAvail\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].availability=int(d[j])
            elif "release\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_aide):
                    instance.list_aide[j].time_window_start[0]=int(d[j])
                for j in range(number_patient):
                    instance.list_patient[j].time_window[0]=int(d[j+number_aide])
                for j in range(number_aide):
                    instance.list_aide[j].time_window_end[0]=int(d[j+number_aide+number_patient])
                                
            elif "deadline\n" in l:
                            
                d=np.array(next(file).split(" "))
                for j in range(number_aide):
                    instance.list_aide[j].time_window_start[1]=int(d[j])
                for j in range(number_patient):
                    instance.list_patient[j].time_window[1]=int(d[j+number_aide])
                for j in range(number_aide):
                    instance.list_aide[j].time_window_end[1]=int(d[j+number_aide+number_patient])
            elif "duration\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].visit_duration=int(d[j+number_aide])
            elif "dayReq\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].number_visit=int(d[j+number_aide])
            elif "code\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].code_value=int(d[j+number_aide])
            elif "code" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].code_value=int(d[j+number_aide])             
            elif "nurseType\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_aide):
                    instance.list_aide[j].type_value=int(d[j])
            elif "maxWeekly\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_aide):
                    instance.list_aide[j].maximum_work_time=int(d[j])
                                
            elif "base\n" in l:
                d=np.array(next(file).split(" "))
                i=0
                for j in range(number_aide):
                    z=j+i
                    instance.list_aide[j].location_start=int(d[z])-1
                    instance.list_aide[j].location_end=int(d[z+1])-1
                    i+=1
                for j in range(number_patient):
                    instance.list_patient[j].location=j+number_aide
            elif "travelTime\n" in l:
                travel=[]
                for i in range(number_node):
                    d=np.array(next(file).split(" "))
                    t=[]
                    for j in range(number_node):
                        t.append(int(d[j]))
                    travel.append(t)
                instance.distance_matrix(travel)
            elif "preAssignNurse\n" in l:
                d=np.array(next(file).split(" "))
                for j in range(number_patient):
                    instance.list_patient[j].pre_assigned_aide=int(d[j])-1
            elif "preAssignDays\n" in l:
                for j in range(number_patient):
                    d=np.array(next(file).split(" "))
                    t=[]
                    for z in range(number_day):
                        t.append(int(d[z]))
                    instance.list_patient[j].pre_assigned_day=t
            elif "preAssignWindows\n" in l:
                for j in range(number_patient):
                    d=np.array(next(file).split(" "))
                    t=(int(d[0]),int(d[1]))
                    instance.list_patient[j].pre_assigned_time_window=t
                    instance.list_patient[j].visit_time=t[0]

