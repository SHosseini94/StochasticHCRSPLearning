# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:13:00 2024

@author: saeed
"""

from Patient import Patient
from Aide import Aide

"""
It saves all the information related to an instance. In an instance, the number of new patients and available aides are determined. 
    The information contains:
        1. instance _name: the name of instance presents the number of new patients in this instance 
        2. number patient: the total number of patients 
        3. number aide: the total number of aides
        4. horizon: the number of planning days 
        5. number node: the summation of number of patients+ 2* number aides 
        6. list patient: this list keeps all the patients information, which means each item in the list associates with one patient object from Patient Class 
        7. list aide: this list keeps all the aides information, which means each item in the list associates with one aide object from Aide Class 
        8. list feasible per aide: a list for each aide keeps the list of all feasible patients set assigned to it, each feasible assignment is a kind of string contain the id of patients assigned to it, for example 23-24-25 could be a record of this list 
        9. list feasible day per aide: a list for each aide keeps the list of all feasible patients set assigned to it for each day, each feasible assignment is a kind of string contain the id of patients assigned to it and also the id of day, for example 23-24-25-/0
        10. distance_matrix: a squared matrix of distances between nodes 

"""
class Instance():
    
    list_patient=[]
    list_aide=[]
    list_available_patient=[]
    list_fixed_patient=[]
    list_free_patient=[]
    list_fixed_aide=[]
    list_free_aide=[]
    
    
    
    def __init__(self, name):
        
        #print("instance created")
        self.instance_name=name
        
    def initilize(self,pnum,anum,nday):
        
        #print("the instance initilization:")
        
        #we set the number of patients, aides, days,and nodes
        self.number_patient=pnum
        self.number_aide=anum
        self.horizon=nday
        self.number_node=self.number_patient+2*self.number_aide
        self.list_patient=[Patient(p) for p in range(self.number_patient)]
        self.list_aide=[Aide(a) for a in range(self.number_aide)]
        self.list_feasible_per_aide=[[]for i in range(self.number_aide)]
        self.list_feasible_day_per_aide=[[]for i in range(self.number_aide)]
        self.list_route=[]
        self.list_schedule=[]
        self.list_visit_count=[]
        self.dic_work_time_day_per_aide=[{}for i in range(self.number_aide)]
        #self.list_infeasible_per_aide=[[]for i in range(self.number_aide)]
        #self.list_infeasible_day_per_aide=[{}for i in range(self.number_aide)]
        
    def distance_matrix(self,travel):
        
        #print("the instance distance_matrix is set")
        self.distance_matrix=travel
        
                