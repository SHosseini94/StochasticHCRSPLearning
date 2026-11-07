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



if __name__=="__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    configure_logging(BASE_DIR)
    logger = logging.getLogger(__name__)

    config_path = os.path.join(BASE_DIR, 'configuration')

    for config_file in os.listdir(config_path):
        if config_file.endswith(".ini"):
            config_path_full= os.path.join(config_path, config_file)

            config = configparser.ConfigParser()
            config.read(config_path_full)

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
            normal_approx_method= config['PARAMETERS']['normal_approx_method']
            deterministic_service= ast.literal_eval(config['PARAMETERS']['deterministic_service'])

            # Dynamic multi-run: instance_files lists one instance per run in order.
            # Each run inherits the accepted patients and visit history from the previous run.
            # scenario_instance is a single file used to build the simulation scenario once.
            instance_files= ast.literal_eval(config['PARAMETERS']['instance_files'])
            scenario_instance= config['PARAMETERS']['scenario_instance']

            data_path= os.path.join(BASE_DIR, 'data', case)
            result_path= os.path.join(BASE_DIR, 'result')

            PROJECT_ROOT = os.path.dirname(BASE_DIR)
            real_data=os.path.join(PROJECT_ROOT,'data-analysis','cleaned_data.csv')

            os.makedirs(data_path, exist_ok=True)

            # Build simulation scenario once from the designated scenario instance.
            # This captures stochastic travel/service properties shared across all runs.
            scenario_file_path = os.path.join(data_path, scenario_instance)
            with open(scenario_file_path, 'r') as file:
                instance_scen=Instance(scenario_instance)
                instance_reader_scen=Instance_Reader()
                instance_reader_scen.read_instance(file, instance_scen)
                logger.info("simulation type: %s", simulation_type)
                scenario_simulation=create_scenario(instance_scen,simulation_scenario,distribution,travel_cov,simulation_type,simulation_mean_learn,simulation_std_learn)


            with Env(empty=True) as env:
                env.setParam('LogToConsole',0)
                env.start()

                # past_aide_visits carries accepted patient history across runs.
                # Format: {patient_id: {aide_id: cumulative_visit_count}}
                past_aide_visits = {}

                # returning_patients: set of patient IDs accepted in the previous run (no-learn case).
                returning_patients = set()

                # past_nolearn_visits: cumulative visits per patient across all previous runs (no-learn).
                # Format: {patient_id: total_visits_before_this_run}
                past_nolearn_visits = {}

                # prev_day_assignment carries the exact per-day aide assignment from the last run.
                # Format: {patient_id: {day: aide_id}} — used as MIP warm start for returning patients.
                prev_day_assignment = {}

                # run_assignments[run] = (inst_file, {patient_id: {aide_id: visits_this_run}})
                # collected incrementally; CoC file is rewritten after each run
                run_assignments = []

                coc_result_path = os.path.join(BASE_DIR, 'result',
                                               'learn' if learning else 'no-learn')
                coc_file_path = os.path.join(coc_result_path, f"dynamic_CoC_{case}.txt")

                for run, inst_file in enumerate(instance_files):

                    file_path = os.path.join(data_path, inst_file)

                    with open(file_path,'r') as file:

                        print("Instance: {} | Run: {}/{}".format(inst_file, run+1, len(instance_files)))
                        instance=Instance(inst_file)
                        instance_reader=Instance_Reader()
                        instance_reader.read_instance(file,instance)

                        # put zero distance from patients to care workers' homes
                        for m in range(instance.number_aide,instance.number_node-instance.number_aide):
                            for n in range(instance.number_node-instance.number_aide,instance.number_node):
                                instance.distance_matrix[m][n]=0

                        print("total number of visits {}".format(sum([i.number_visit for i in instance.list_patient if i.availability==2])))

                        # Attach prior-run visit counts so subproblems can compute accumulated visits.
                        instance.prior_visit_counts = dict(past_nolearn_visits)

                        run_file_name = (inst_file + '-' + case +
                                         '-run' + str(run) +
                                         '-' + f"{simulation_type}" +'-count-prior-visits-'
                                         '-teamsize-5-')

                        if learning:

                            run_file_name= run_file_name+'gamma1-'+f"{gamma1}" + 'gamma2-' +f"{gamma2}" + '-' +f"{normal_approx_method}"

                            if deterministic_service:
                                pass
                            else:

                                if assignment:

                                    master=Assignment_Master_Problem(run_file_name,instance,overtime,alpha,beta,gamma1,TCov, mean_learn,std_learn,team,team_size,similar_first)

                                    total_time,obj=master.branch_and_check(env)
                                else:

                                    master=Master_Problem(run_file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov, mean_learn,std_learn,team,team_size,similar_first,
                                                          past_aide_visits=past_aide_visits,
                                                          prev_day_assignment=prev_day_assignment)

                                    total_time,obj=master.branch_and_check(env)
                                    #r,obj= master.logic_based_benders(env)
                        else:

                            if deterministic_service:
                                run_file_name= run_file_name + '-' +'deterministic'+'-' +f"{mean_no_learn}"

                                master=Deterministic_No_Learn_Master_Problem(run_file_name,instance,overtime,TCov,team,team_size,mean_no_learn)
                                total_time,obj=master.branch_and_check(env)

                            else:
                                run_file_name= run_file_name + 'gamma1-'+f"{gamma1}" + 'gamma2-' +f"{gamma2}" + '-' +f"{mean_no_learn}" + '-' +f"{std_no_learn}"
                                if assignment:

                                    master=Assignment_No_Learn_Master_Problem(run_file_name,instance,overtime,alpha,beta,gamma1,TCov, team,team_size,mean_no_learn,std_no_learn)

                                    total_time,obj=master.branch_and_check(env)
                                else:

                                    master=No_Learn_Master_Problem(run_file_name,instance,overtime,alpha,beta,gamma1,gamma2,TCov,team,team_size,mean_no_learn,std_no_learn,
                                                                   returning_patients=returning_patients)

                                    total_time,obj=master.branch_and_check(env)


                        logger.info("Run %s — instance: %s", run, run_file_name)

                        if obj==0:
                            print("Learning:{} Run {} incomplete".format(learning, run))
                            logger.warning("Learning:%s Run %s incomplete — stopping dynamic chain", learning, run)
                            # No solution: cannot propagate history to the next run
                            break
                        else:
                            print("Learning:{} Run {} finished".format(learning, run)+run_file_name+" CPU time {}".format(total_time))
                            logger.info("Learning (Deterministic:%s):%s Run %s finished %s CPU time %s", deterministic_service, learning, run, run_file_name, total_time)

                        # No-learn: carry accepted patient IDs to next run (delta fix only).
                        if not learning and not assignment and not deterministic_service:
                            returning_patients = {pat.id for pat in instance.list_patient
                                                  if pat.availability==2 and pat.aide_assigned}
                            logger.info("Run %s — %s returning patients for no-learn next run",
                                        run, len(returning_patients))

                            # Accumulate per-(aide, patient) visit counts for simulation's prior_visit_counts.
                            for pat in instance.list_patient:
                                if pat.availability == 2 and pat.aide_assigned:
                                    if pat.id not in past_nolearn_visits:
                                        past_nolearn_visits[pat.id] = {}
                                    for aide_id, count in pat.aide_assigned.items():
                                        past_nolearn_visits[pat.id][aide_id] = past_nolearn_visits[pat.id].get(aide_id, 0) + count

                            # Store this run's assignments and rewrite the CoC comparison table.
                            new_visits = Master_Problem.extract_past_visits(instance)
                            run_assignments.append((inst_file, new_visits))

                            all_patients = sorted(set(p for _, v in run_assignments for p in v))
                            col_w = 22

                            with open(coc_file_path, 'w') as coc_f:
                                coc_f.write("Dynamic Continuity of Care\n")
                                coc_f.write("Case: {}\n".format(case))
                                coc_f.write("Format: Aide:visits (this run) per patient per run. '-' = not accepted.\n")
                                coc_f.write("=" * (10 + col_w * len(run_assignments)) + "\n\n")

                                coc_f.write("{:<10}".format("Patient"))
                                for r, (inst, _) in enumerate(run_assignments):
                                    label = "Run{}({})".format(r, inst.split("patient-")[1].split("-aide")[0]+"pt")
                                    coc_f.write("{:<{}}".format(label, col_w))
                                coc_f.write("\n")
                                coc_f.write("-" * (10 + col_w * len(run_assignments)) + "\n")

                                for pat_id in all_patients:
                                    coc_f.write("{:<10}".format(pat_id))
                                    for r, (_, visits) in enumerate(run_assignments):
                                        if pat_id in visits:
                                            cell = " ".join("A{}:{}v".format(a, c)
                                                            for a, c in sorted(visits[pat_id].items()))
                                        else:
                                            cell = "-"
                                        coc_f.write("{:<{}}".format(cell, col_w))
                                    coc_f.write("\n")

                                coc_f.write("\nSummary (same aide kept from previous run)\n")
                                coc_f.write("-" * 50 + "\n")
                                for r in range(1, len(run_assignments)):
                                    _, prev = run_assignments[r - 1]
                                    _, curr = run_assignments[r]
                                    returning = [p for p in curr if p in prev]
                                    consistent = sum(
                                        1 for p in returning
                                        if any(a in prev[p] for a in curr[p])
                                    )
                                    coc_f.write("Run {}: {}/{} returning patients kept at least one past aide\n".format(
                                        r, consistent, len(returning)))

                        # Learning: accumulate visit history and carry exact day assignment.
                        # Only Master_Problem (learning, non-assignment) supports these.
                        if learning and not assignment and not deterministic_service:
                            prev_day_assignment = master.result_day_assignment
                            new_visits = Master_Problem.extract_past_visits(instance)
                            for pat_id, aide_dict in new_visits.items():
                                if pat_id not in past_aide_visits:
                                    past_aide_visits[pat_id] = {}
                                for aide_id, count in aide_dict.items():
                                    past_aide_visits[pat_id][aide_id] = past_aide_visits[pat_id].get(aide_id, 0) + count
                            logger.info("Run %s — %s patients carry cumulative visit history into run %s",
                                        run, len(past_aide_visits), run+1)

                            # Store this run's assignments and rewrite the full comparison table
                            run_assignments.append((inst_file, new_visits))

                            all_patients = sorted(set(p for _, v in run_assignments for p in v))
                            col_w = 22

                            with open(coc_file_path, 'w') as coc_f:
                                coc_f.write("Dynamic Continuity of Care\n")
                                coc_f.write("Case: {}\n".format(case))
                                coc_f.write("Format: Aide:visits (this run) per patient per run. '-' = not accepted.\n")
                                coc_f.write("=" * (10 + col_w * len(run_assignments)) + "\n\n")

                                # Header row: one column per run
                                coc_f.write("{:<10}".format("Patient"))
                                for r, (inst, _) in enumerate(run_assignments):
                                    label = "Run{}({})".format(r, inst.split("patient-")[1].split("-aide")[0]+"pt")
                                    coc_f.write("{:<{}}".format(label, col_w))
                                coc_f.write("\n")
                                coc_f.write("-" * (10 + col_w * len(run_assignments)) + "\n")

                                # One row per patient
                                for pat_id in all_patients:
                                    coc_f.write("{:<10}".format(pat_id))
                                    for r, (_, visits) in enumerate(run_assignments):
                                        if pat_id in visits:
                                            cell = " ".join("A{}:{}v".format(a, c)
                                                            for a, c in sorted(visits[pat_id].items()))
                                        else:
                                            cell = "-"
                                        coc_f.write("{:<{}}".format(cell, col_w))
                                    coc_f.write("\n")

                                # Summary: continuity rate per run (vs run 0)
                                coc_f.write("\nSummary (same aide kept from previous run)\n")
                                coc_f.write("-" * 50 + "\n")
                                for r in range(1, len(run_assignments)):
                                    _, prev = run_assignments[r - 1]
                                    _, curr = run_assignments[r]
                                    returning = [p for p in curr if p in prev]
                                    consistent = sum(
                                        1 for p in returning
                                        if any(a in prev[p] for a in curr[p])    
                                    )
                                    coc_f.write("Run {}: {}/{} returning patients kept at least one past aide\n".format(
                                        r, consistent, len(returning)))

                        print("list visit count")
                        print(instance.list_visit_count)
                        simulate=simulation()
                        average_total_traveltime,average_total_waitingtime,average_total_lateness,average_total_overtime,average_run_lateness,average_run_overtime,max_total_lateness,patient_service_record,patient_mean_service,patient_min_service,visit_service_record,visit_mean_service,visit_min_service,patient_below_alpha,visit_below_alpha,aide_service_record,aide_min_service,aide_mean_service,aide_below_beta=simulate.total_delay(instance,scenario_simulation,overtime,0.95,0.95)


                        file_path_result=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
                        if learning:
                            file_path_result=os.path.join(file_path_result,'learn')
                        else:
                            if deterministic_service:
                                file_path_result=os.path.join(file_path_result,'no-learn','deterministic')
                            else:
                                file_path_result=os.path.join(file_path_result,'no-learn')

                        with open(os.path.join(file_path_result,"{}_learning_{}_alpha_{}_beta_{}_distribution_{}_{}_approximation_analysis.txt".format(run_file_name,learning,alpha,beta,distribution,run)),"w") as f:

                                    f.write("\n")
                                    f.write("Method\t"+"EVT \n")
                                    f.write("Run\t{}\n".format(run))
                                    f.write("InstanceFile\t{}\n".format(inst_file))
                                    f.write("ReturnPatients\t{}\n".format(len(past_aide_visits)))
                                    f.write("Learning\t{}\n".format(learning))
                                    if learning:
                                        f.write("MeanLearn\t{}\n".format(mean_learn))
                                        f.write("StdLearn\t{}\n".format(std_learn))
                                    else:
                                        if deterministic_service:
                                            f.write("MeanNoLearn\t{}\n".format(mean_no_learn))
                                        else:
                                            f.write("MeanNoLearn\t{}\n".format(mean_no_learn))
                                            f.write("StdNoLearn\t{}\n".format(std_no_learn))
                                    f.write("Data Case\t{}\n".format(case))
                                    f.write("Assignment\t{}\n".format(assignment))
                                    f.write("NumSimulationScenario\t{} \n".format(simulation_scenario))
                                    f.write("SimulationType\t{} \n".format(simulation_type))
                                    f.write("Distribution\t"+distribution + "\n")
                                    f.write("TravelCov\t{} \n".format(travel_cov))
                                    f.write("ServiceCov\t{} \n".format(service_cov))
                                    f.write("NormalApproxMethod\t{} \n".format(normal_approx_method))

                                    f.write("NewPatient\t{} \n".format(len([1 for i in range(instance.number_patient) if instance.list_patient[i].availability==2 ])))

                                    f.write("ActiveAides\t{} \n".format(len([1  for i in range(instance.number_aide) if instance.list_aide[i].availability==2])))
                                    f.write("TotalVisitRequested\t{} \n".format(sum([i.number_visit for i in instance.list_patient if i.availability==2])))

                                    f.write("TotalSolvingTime\t{}\n".format(total_time))
                                    f.write("TotalAcceptedVisits\t{}\n".format(obj))

                                    f.write("Gamma1\t{}\n".format(gamma1))
                                    f.write("Gamma2\t{}\n".format(gamma2))
                                    f.write("Overtime\t{}\n".format(overtime))
                                    f.write("PatientAssignedServiceLevel/NumberStd\t{}\n".format(alpha))
                                    f.write("AideAssignedServiceLevel/NumberStd\t{}\n".format(beta))
                                    f.write("Team\t{}\n".format(team))
                                    f.write("TeamSize\t{}\n".format(team_size))
                                    f.write("ApproximationTCov\t{}\n".format(TCov))
                                    f.write("ApproximationSCov\t{}\n".format(SCov))

                                    f.write("AverageTotalTravelTime\t{}\n".format(average_total_traveltime))
                                    f.write("AverageTotalWaitingTime\t{}\n".format(average_total_waitingtime))
                                    f.write("AverageTotalLateness\t{}\n".format(average_total_lateness))
                                    f.write("AverageTotalOvertime\t{}\n".format(average_total_overtime))
                                    f.write("AverageRunLateness\t{}\n".format(average_run_lateness))
                                    f.write("AverageRunOvertime\t{}\n".format(average_run_overtime))
                                    f.write("MaximumTotalLateness\t{}\n".format(max_total_lateness))

                                    f.write("PatientServiceLevel\t")
                                    patient_service_level=[(i,float(v)) for i,v in patient_service_record]
                                    f.write(str(patient_service_level))
                                    f.write("\n")

                                    f.write("PatientMeanServiceLevel\t{}\n".format(patient_mean_service))
                                    f.write("PatientMinServiceLevel\t{}\n".format(patient_min_service))

                                    f.write("VisitServiceLevel\t")
                                    visit_service_level=[(i,[float(v[j]) for j in range(len(v))]) for i,v in visit_service_record]
                                    f.write(str(visit_service_level))
                                    f.write("\n")

                                    f.write("VisitMeanServiceLevel\t{}\n".format(visit_mean_service))
                                    f.write("VisitMinServiceLevel\t{}\n".format(visit_min_service))

                                    f.write("RouteServiceLevel\t")
                                    f.write("\n")
                                    route_service_level=[(i,[float(v[j]) for j in range(len(v))]) for i,v in aide_service_record]
                                    f.write(str(route_service_level))
                                    f.write("\n")

                                    f.write("RouteMeanServiceLevel\t{}\n".format(aide_mean_service))
                                    f.write("RouteMinServiceLevel\t{}\n".format(aide_min_service))

                                    f.write("NumAcceptedPatient:NumBelowAlpha\t{}\n".format(patient_below_alpha))
                                    f.write("NumAccpetedVisit:NumBelowAlpha\t{}\n".format(visit_below_alpha))
                                    f.write("NumRoutes:NumBelowBeta\t{}\n".format(aide_below_beta))
