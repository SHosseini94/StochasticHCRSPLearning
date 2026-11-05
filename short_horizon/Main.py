# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:30:34 2024

@author: saeed
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:05:42 2024

@author: saeed
"""

from Instance import Instance
from Instance_Reader import Instance_Reader
from Simulation import simulation
from Create_Scenario import create_scenario
from Master_Problem import Master_Problem
from Master_Problem import No_Learn_Master_Problem
from Master_Problem import Assignment_Master_Problem
from Master_Problem import Assignment_No_Learn_Master_Problem
from Master_Problem import Deterministic_No_Learn_Master_Problem
from Master_Problem import Deterministic_Learn_Master_Problem
from gurobipy import *
from logging_setup import configure_logging

import os 
import sys 
import getopt
import numpy as np
import ast
import configparser
import argparse
import logging 
import pandas as pd



if __name__=="__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    configure_logging(BASE_DIR)
    logger = logging.getLogger(__name__)
    
    #parser= argparse.ArgumentParser(description="Run the main script with a specific configuration file.")
    #parser.add_argument("--config", type=str, required= True, help="Path to the configuration file (.ini.format).")
    #args= parser.parse_args()

    
    #config_path = os.path.join(BASE_DIR,'config.ini')
    config_path = os.path.join(BASE_DIR, 'configuration')
    #print("The configuration file path is: "+config_path)
    #config_path= args.config

    for config_file in os.listdir(config_path):
        if config_file.endswith(".ini"):
            config_path_full= os.path.join(config_path, config_file)
            #print("The configuration file used is: "+config_path_full)


            config = configparser.ConfigParser()
            config.read(config_path_full)

            #delay= int(config['PARAMETERS']['delay'])
            overtime= int(config['PARAMETERS']['overtime'])
            alpha= float(config['PARAMETERS']['alpha'])
            beta= float(config['PARAMETERS']['beta'])
            gamma1= float(config['PARAMETERS']['gamma1'])
            gamma2= float(config['PARAMETERS']['gamma2'])
            TCov= float(config['PARAMETERS']['TCov'])
            SCov= float(config['PARAMETERS']['SCov'])
            distribution= config['PARAMETERS']['distribution']
            travel_cov= float(config['PARAMETERS']['travel_cov'])
            service_cov= float(config['PARAMETERS']['service_cov'])
            #arg_coeff= ast.literal_eval(config['PARAMETERS']['coeff'])
            #coeff=np.array(arg_coeff)  #[rainy,clear]
            #arg_prob=ast.literal_eval(config['PARAMETERS']['prob'])
            #prob=np.array(arg_prob)
            #correlation=config['PARAMETERS']['correlation']
            mean_learn= ast.literal_eval(config['PARAMETERS']['mean_learn'])
            std_learn= ast.literal_eval(config['PARAMETERS']['std_learn'])
            mean_no_learn= ast.literal_eval(config['PARAMETERS']['mean_no_learn'])
            std_no_learn= ast.literal_eval(config['PARAMETERS']['std_no_learn'])
            simulation_scenario=int(config['PARAMETERS']['simulation_scenario'])
            simulation_type= config['PARAMETERS']['simulation_type']
            simulation_mean_learn= ast.literal_eval(config['PARAMETERS']['simulation_mean_learn'])
            simulation_std_learn= ast.literal_eval(config['PARAMETERS']['simulation_std_learn'])

            iteration= int(config['PARAMETERS']['iteration'])
            case= config['PARAMETERS']['case']
            team=config['PARAMETERS']['team']
            team_size=int(config['PARAMETERS']['team_size'])
            similar_first= ast.literal_eval(config['PARAMETERS']['similar_first'])
            learning= ast.literal_eval(config['PARAMETERS']['learning'])
            assignment= ast.literal_eval(config['PARAMETERS']['assignment'])
            reduction= ast.literal_eval(config['PARAMETERS']['reduction'])  
            normal_approx_method= config['PARAMETERS']['normal_approx_method'] # standard: find mu and sigma based on simple mean and std for each cum visit group , percentile: find mu and sigma to fit alpha percentile requirement
            deterministic_service= ast.literal_eval(config['PARAMETERS']['deterministic_service'])

            data_path= os.path.join(BASE_DIR, 'data', case)
            result_path= os.path.join(BASE_DIR, 'result')

            PROJECT_ROOT = os.path.dirname(BASE_DIR) # where EVT-based folder is located; chance-constraint 
            real_data=os.path.join(PROJECT_ROOT,'data-analysis','cleaned_data.csv')
            #print("The data path is: "+data_path)
            #print("The result path is: "+result_path)

            
            os.makedirs(data_path, exist_ok=True)
                
            # read default instance file and create scenarios for simulation 

            
            for file_name in os.listdir(data_path):
                if file_name.endswith("client-30-caregiver-6.txt"):
                    file_path = os.path.join(data_path, file_name)
                    with open(file_path,'r') as file:
                        #print("the instance name is "+f)
                        instance=Instance(file_name)
                        instance_reader=Instance_Reader()
                        instance_reader.read_instance(file,instance)
                        #instance_reader.print_instance(instance)
                        logger.info("simulation type: %s", simulation_type)
                        scenario_simulation=create_scenario(instance,simulation_scenario,distribution,travel_cov,simulation_type,simulation_mean_learn,simulation_std_learn)
                        

            
            with Env(empty=True) as env: # create a new Gurobi environment 
                #env.setParam('OutputFlag',0)  # turn off the Gurobi output 
                env.setParam('LogToConsole',0)  # turn off the Gurobi output to console
                env.start() # start the environment
            
                for i in range(iteration):
                    
                    for file_name in os.listdir(data_path):
                        if file_name.endswith(".txt"):
                            #print("the instance name is "+f)
                            

                            file_path = os.path.join(data_path, file_name)

                            with open(file_path,'r') as file:

                                print("the instance name is "+file_name)
                                instance=Instance(file_name)
                                instance_reader=Instance_Reader()
                                instance_reader.read_instance(file,instance)

                                

                                # put zero distance from patients to care workers' homes
                                for m in range(instance.number_aide,instance.number_node-instance.number_aide):
                                    for n in range(instance.number_node-instance.number_aide,instance.number_node):
                                        instance.distance_matrix[m][n]=0

                                print("total number of visits {}".format(sum([i.number_visit for i in instance.list_patient if i.availability==2])))
                                
                                file_name= file_name + '-' + case + '-' +f"{i}"+ '-'+f"{simulation_type}"+'-'                           
                                


                                
                                if learning:

                                    if deterministic_service:
                                        file_name= file_name + '-' +'deterministic'+'-' +'Learning-'

                                        master=Deterministic_Learn_Master_Problem(file_name,instance,overtime,TCov,mean_learn,team,team_size)
                                        r,obj=master.branch_and_check(env)
                                    else:
                                        file_name= file_name+'gamma1-'+f"{gamma1}" + 'gamma2-' +f"{gamma2}" + '-' 
                                        if assignment:

                                            master=Assignment_Master_Problem(file_name,instance,overtime,alpha,beta,gamma1,TCov, mean_learn,std_learn,team,team_size,similar_first)

                                            r,obj=master.branch_and_check(env)
                                        else:

                                            master=Master_Problem(file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov, mean_learn,std_learn,team,team_size,similar_first)

                                            r,obj=master.branch_and_check(env)
                                            #r,obj= master.logic_based_benders(env)
                                else:
                                    

                                    if deterministic_service:
                                        file_name= file_name + '-' +'deterministic'+'-' +f"{mean_no_learn}" 
                                        

                                        master=Deterministic_No_Learn_Master_Problem(file_name,instance,overtime,TCov,team,team_size,mean_no_learn)
                                        r,obj=master.branch_and_check(env)

                                    else:
                                        file_name= file_name + 'gamma1-'+f"{gamma1}" + 'gamma2-' +f"{gamma2}" + '-' +f"{mean_no_learn}" + '-' +f"{std_no_learn}"
                                        if assignment:

                                            master=Assignment_No_Learn_Master_Problem(file_name,instance,overtime,alpha,beta,gamma1,TCov, team,team_size,mean_no_learn,std_no_learn)

                                            r,obj=master.branch_and_check(env)
                                        else:

                                            master=No_Learn_Master_Problem(file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov,team,team_size,mean_no_learn,std_no_learn)

                                            r,obj=master.branch_and_check(env)


                                logger.info("Processing instance file: %s", file_name)

                                if obj==0:
                                    print("Learning:{} The problem is incomplete".format(learning))
                                    logger.warning("Learning:%s The problem is incomplete", learning)
                                else:
                                    print("Learning:{} Finished".format(learning)+file_name+" with CPU time {}".format(r))
                                    logger.info("Learning (Deterministic:%s):%s Finished %s with CPU time %s", learning,deterministic_service, file_name, r)
                                
                                #print("the feasible patterns for aide 0 :")
                                #print(instance.list_feasible_day_per_aide[0])

                                #print("the feasible patterns for aide 2 :")
                                #print(instance.list_feasible_day_per_aide[2])

                                
                                simulate=simulation()
                                average_total_traveltime,average_total_waitingtime,average_total_lateness,average_total_overtime,average_run_lateness,average_run_overtime,max_total_lateness,patient_service_record,patient_mean_service,patient_min_service,visit_service_record,visit_mean_service,visit_min_service,patient_below_alpha,visit_below_alpha,aide_service_record,aide_min_service,aide_mean_service,aide_below_beta,record_lateness_overtime=simulate.total_delay(instance,scenario_simulation,overtime,0.95,0.95)
                                

                                file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
                                if learning:
                                    if deterministic_service:
                                        file_path=os.path.join(file_path,'learn','deterministic')
                                    else:
                                        file_path=os.path.join(file_path,'learn')
                                else:
                                    if deterministic_service:
                                        file_path=os.path.join(file_path,'no-learn','deterministic')
                                    else:
                                        file_path=os.path.join(file_path,'no-learn')
                                with open(os.path.join(file_path,"{}_learning_{}_alpha_{}_beta_{}_distribution_{}_{}_approximation_analysis.txt".format(file_name,learning,alpha,beta,distribution,i)),"w") as file:
                                
                                            file.write("\n")
                                            file.write("Method\t"+"EVT \n")
                                            file.write("Learning\t{}\n".format(learning))
                                            if learning:
                                                file.write("MeanLearn\t{}\n".format(mean_learn))
                                                if not deterministic_service:
                                                    file.write("StdLearn\t{}\n".format(std_learn))
                                            else:
                                                if deterministic_service:
                                                    file.write("MeanNoLearn\t{}\n".format(mean_no_learn))
                                                else:
                                                    file.write("MeanNoLearn\t{}\n".format(mean_no_learn))
                                                    file.write("StdNoLearn\t{}\n".format(std_no_learn))
                                            #file.write("DeterministicService\t{}\n".format(deterministic_service))
                                            file.write("Data Case\t{}\n".format(case))
                                            file.write("Assignment\t{}\n".format(assignment))
                                            file.write("NumSimulationScenario\t{} \n".format(simulation_scenario))
                                            file.write("SimulationType\t{} \n".format(simulation_type))
                                            file.write("Distribution\t"+distribution + "\n")
                                            #file.write("FixedVisit\t{} \n".format(fixed_visit))
                                            file.write("TravelCov\t{} \n".format(travel_cov))
                                            #file.write("TravelCovMax\t{} \n".format(travel_cov_max))
                                            file.write("ServiceCov\t{} \n".format(service_cov))
                                            file.write("NormalApproxMethod\t{} \n".format(normal_approx_method))



                                            file.write("NewPatient\t{} \n".format(len([1 for i in range(instance.number_patient) if instance.list_patient[i].availability==2 ])))

                                            file.write("ActiveAides\t{} \n".format(len([1  for i in range(instance.number_aide) if instance.list_aide[i].availability==2])))
                                            file.write("TotalVisitRequested\t{} \n".format(sum([i.number_visit for i in instance.list_patient if i.availability==2])))

                                            file.write("TotalSolvingTime\t{}\n".format(r))
                                            file.write("TotalAcceptedVisits\t{}\n".format(obj))

                                            file.write("Gamma1\t{}\n".format(gamma1))
                                            file.write("Gamma2\t{}\n".format(gamma2))
                                            file.write("Overtime\t{}\n".format(overtime))
                                            file.write("PatientAssignedServiceLevel/NumberStd\t{}\n".format(alpha))
                                            file.write("AideAssignedServiceLevel/NumberStd\t{}\n".format(beta))
                                            file.write("Team\t{}\n".format(team))
                                            file.write("TeamSize\t{}\n".format(team_size))
                                            #file.write("Scenario/Cov\t{}\n".format(TCov))
                                            #file.write("Scenario\t{}\n".format(0))
                                            file.write("ApproximationTCov\t{}\n".format(TCov))
                                            file.write("ApproximationSCov\t{}\n".format(SCov))
                                        
                                            file.write("AverageTotalTravelTime\t{}\n".format(average_total_traveltime))
                                            file.write("AverageTotalWaitingTime\t{}\n".format(average_total_waitingtime))
                                            file.write("AverageTotalLateness\t{}\n".format(average_total_lateness))
                                            file.write("AverageTotalOvertime\t{}\n".format(average_total_overtime))
                                            file.write("AverageRunLateness\t{}\n".format(average_run_lateness))
                                            file.write("AverageRunOvertime\t{}\n".format(average_run_overtime))
                                            file.write("MaximumTotalLateness\t{}\n".format(max_total_lateness))


                                            file.write("PatientServiceLevel\t")
                                            patient_service_level=[(i,float(v)) for i,v in patient_service_record]
                                            
                                            file.write(str(patient_service_level))
                                            file.write("\n")

                                            file.write("PatientMeanServiceLevel\t{}\n".format(patient_mean_service))
                                            file.write("PatientMinServiceLevel\t{}\n".format(patient_min_service))

                                            file.write("VisitServiceLevel\t")
                                            visit_service_level=[(i,[float(v[j]) for j in range(len(v))]) for i,v in visit_service_record]
                                            
                                            file.write(str(visit_service_level))
                                            file.write("\n")

                                            file.write("VisitMeanServiceLevel\t{}\n".format(visit_mean_service))
                                            file.write("VisitMinServiceLevel\t{}\n".format(visit_min_service))

                                            file.write("RouteServiceLevel\t")
                                            file.write("\n")
                                            route_service_level=[(i,[float(v[j]) for j in range(len(v))]) for i,v in aide_service_record]  
                                            file.write(str(route_service_level))
                                            file.write("\n")
                                            
                                            file.write("RouteMeanServiceLevel\t{}\n".format(aide_mean_service))
                                            file.write("RouteMinServiceLevel\t{}\n".format(aide_min_service))

                                            file.write("NumAcceptedPatient:NumBelowAlpha\t{}\n".format(patient_below_alpha))
                                            file.write("NumAccpetedVisit:NumBelowAlpha\t{}\n".format(visit_below_alpha))
                                            file.write("NumRoutes:NumBelowBeta\t{}\n".format(aide_below_beta))

                                #comment for now 
                                df= pd.DataFrame({k:pd.Series(v) for k,v in record_lateness_overtime.items()})
                                df.to_csv(os.path.join(file_path,"{}_learning_{}_alpha_{}_beta_{}_{}_lateness_overtime.csv".format(file_name,learning,alpha,beta,i)), index=False)
                                                                    
                            
                                    
            
                       
                                
                                
            
            


