# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:49:01 2024

@author: saeed
"""

"""
It saves all the information related to an aide.
    The information contains:
        1. id: the id number of this aide. 
        2. location_start: the node id of the aide's start location (the nodes start with the aides' start location, then patients, and finally the aide's end location). 
        3. location_end: the node id of the aide's end location (the nodes start with the aides' start location, then patient, and finally the aide's end location). 
        4. maximum_work_time: the maximum allowed work time during a week.
        5. availability: the status of an aide in the system: 1: not available for new patients 2: available for new patients 
        6. type_value: aides type value, for this instances all the aides are the same type.
        7. time_window_start: the earliest start time for this aide 
        8. time_window_end: the latest end time for this aide 
        9. route: the order of patients visited by this aide in each day 
"""

class Aide():
    def __init__(self, aide_id):
        self.id=aide_id
        self.location_start=-1
        self.location_end=-1
        self.maximum_work_time=-1
        self.availability=-1
        self.type_value=-1
        self.time_window_start=[-1,-1]
        self.time_window_end=[-1,-1]
        #self.route=[[] for d in range(5)]
        self.route=[]
        self.schedule=[]
        self.visit_count=[]