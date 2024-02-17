# -*- coding: utf-8 -*-
__author__ = "Zhongwang Wei / zhongwang007@gmail.com"
__version__ = "0.1"
__release__ = "0.1"
__date__ = "Mar 2023"
import sys,os
from namelist_read import namelist_read
from namelist_read import get_general_info
#from Makefiles_parallel import Makefiles_parallel
from Validation import Validation_geo,Validation_stn
from Data_handle import DatasetHandler
from Figure_handle import FigureHandler
os.environ['PYTHONWARNINGS']='ignore::UserWarning'
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'

if __name__=='__main__':
    print("Welcome to the Geo module of the validation system!")
    print("This module is used to validate the Geo information of the model output data")
    print("===============================================================================")
    print("Start running evaluation system...")
    
    print("-------------------------------------Caution-----------------------------------")
    print("Please make sure the time axis of the simulation data is consistent with the time axis of the validation data!")
    #input("Press Enter to continue...")
    print("...............................................................................")
    argv                      = sys.argv
    nml                       = str(argv[1])

    nl = namelist_read()  # Create an instance of the class
    main_nl    =  nl.read_namelist(f'{nml}')
    #read main_nl['Evaluation_Items'], if the Evaluation_Item is set as true, then select the Evaluation_Item from the Evaluation_Items
    evaluation_items    =  nl.select_variables(main_nl['evaluation_items']).keys()
    #read main_nl['metrics'], if the metric is set as true, then select the metric from the metrics
    metric_vars         =  nl.select_variables(main_nl['metrics']).keys()
    #read main_nl['metrics'], if the metric is set as true, then select the metric from the metrics
    score_vars          =  nl.select_variables(main_nl['scores']).keys()
    #read the reference namelist
    ref_nml    =  nl.read_namelist(f'{main_nl["general"]["reference_nml"]}')

    #read the simulation namelist
    sim_nml    =  nl.read_namelist(f'{main_nl["general"]["simulation_nml"]}')

    for evaluation_item in evaluation_items:
        if evaluation_item not in sim_nml.keys():
            print(f"Error: {evaluation_item} is not in the simulation namelist!")
            exit()
        else:
            pass
        
        #read the simulation source and reference source
        sim_sources    =  sim_nml['general'][f'{evaluation_item}_sim_source']
        ref_sources    =  ref_nml['general'][f'{evaluation_item}_ref_source']
        #if the sim_sources and ref_sources are not list, then convert them to list
        if isinstance(sim_sources, str): sim_sources = [sim_sources]
        if isinstance(ref_sources, str): ref_sources = [ref_sources]

        for sim_source in sim_sources:
            for ref_source in ref_sources:
                print("===============================================================================")
                general_info_object=get_general_info(evaluation_item,sim_nml,ref_nml,main_nl,sim_source,ref_source,metric_vars,score_vars)
                general_info = general_info_object.to_dict()
                if main_nl['general']['evaluation_only']:
                    print('evaluation only, no makefiles')
                    pass
                else:
                    k=DatasetHandler(general_info)
                    k.run()

                '''
                    ppp1 = Makefiles_parallel(general_info)
                    ppp1.Makefiles_parallel()
                '''
                if general_info['ref_data_type'] == 'stn' or general_info['sim_data_type'] == 'stn':
                    print('yeh, we are going to run the stn evaluation module!')
                    k1=Validation_stn(general_info)
                    k1.make_validation_P()
                    k1.make_plot_index()
                else:
                    validation = Validation_geo(general_info)
                    validation.make_validation()
                    validation.make_plot_index()
