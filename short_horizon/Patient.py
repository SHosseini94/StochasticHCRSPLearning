# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:02:31 2024

@author: saeed
"""

"""
It saves all the information related to a patient. 
    The information contains: 
        1. id 
        2. number visit: number of required visits during a week 
        3. availability: type of patients: 1: old patient 2: new patient 0: left patient 
        4. visit duration: the duration of a visit 
        5. code value: number of minimum days need to be between visits: [0,1,2]
        6. location: the id of patient node (the nodes start with the aides' start location, then patient, and finally the aide's end location).
        7. time window: [minimum start time, maximum end time]
        8. pre assigned aide: the id of aide who is assigned to this patient. It is practical when the patient is an old one. 
        9. pre assigned day: [Monday, Tuesday, Wednesday, Thursday, Friday] each one has a binary value: 0 if the day is not a visit day, 1 othewirse
        10. pre assigned time window: [start service, end service] the time window assigned to an old patient 
        11. visit_time: the assigned visit time  
"""
class Patient():
    
    def __init__(self, patient_id):
        
        self.id=patient_id
        self.number_visit=-1
        self.availability=-1
        self.visit_duration=-1
        self.mean_visit_duration={}
        self.std_visit_duration={}
        self.code_value=-1
        self.location=-1
        self.time_window=[-1,-1]
        self.pre_assigned_aide=-1
        self.pre_assigned_day=[]
        self.pre_assigned_time_window=()
        self.visit_time=0
        self.aide_assigned={}

