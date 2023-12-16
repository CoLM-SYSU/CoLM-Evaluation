# -*- coding: utf-8 -*-
import numpy as np
import sys
import xarray as xr
import shutil 
import os
import pandas as pd



class namelist_read:
    def __init__(self):
        
        """
        A libray with Python functions for calculations of
        micrometeorological parameters and some miscellaneous
        utilities.
        functions:
        apb                 : absolute percent bias
        rmse                : root mean square error
        mae                 : mean absolute error
        bias                : bias
        pc_bias             : percentage bias
        NSE                 : Nash-Sutcliffe Coefficient
        L                   : likelihood estimation
        correlation         : correlation
        correlation_R2      : correlation**2, R2
        index_agreement     : index of agreement
        KGE                 : Kling-Gupta Efficiency
        
        """
        self.name = 'metrics'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        np.seterr(all='ignore')  

        # Turn off only the RuntimeWarning
        #np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
    def strtobool (self,val):
        """Convert a string representation of truth to true (1) or false (0).
        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))
    def select_variables(self,namelist):
        #select variables from namelist if the value is true
        select_variables = {k: v for k, v in namelist.items() if v}
        return select_variables
    def read_namelist(self,file_path):
        """
        Read a namelist from a text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            dict: A dictionary containing the keys and values from the namelist.
        """
        namelist = {}
        current_dict = None
        with open(file_path, 'r') as f:
            for line in f:
                # Ignore comments (lines starting with '#')
                if line.startswith('#'):
                    continue
                elif not line:
                    continue
                # Ignore if line is empty，or only contains whitespace
                elif not line.strip():
                    continue
                else:
                    # Check if line starts with '&', indicating a new dictionary
                    if line.startswith('&'):
                        dict_name = line.strip()[1:]
                        current_dict = {}
                        namelist[dict_name] = current_dict
                    # Check if line starts with '/', indicating the end of the current dictionary
                    elif line.startswith('/'):
                        current_dict = None
                    # Otherwise, add the key-value pair to the current dictionary
                    elif current_dict is not None:
                        line = line.split("#")[0]
                        if  not line.strip():
                            continue
                        else:
                            try:
                                key, value = line.split('=')
                                current_dict[key.strip()] = value.strip()
                            except ValueError:
                                print(f"Warning: Skipping line '{line}' as it doesn't contain '='")
                            #if the key value in dict is True or False, set its type to bool
                            if value.strip() == 'True' or value.strip() == 'true' or value.strip() == 'False' or value.strip() == 'false':
                                current_dict[key.strip()] = bool(self.strtobool(value.strip()))
                            #if the key str value in dict is a positive or negative int (), set its type to int
                            elif value.strip().isdigit() or value.strip().replace('-','',1).isdigit():
                                current_dict[key.strip()] = int(value.strip())
                            #if the key str value in dict is a positive or negative float (), set its type to int or float
                            elif value.strip().replace('.','',1).isdigit() or value.strip().replace('.','',1).replace('-','',1).isdigit():
                                current_dict[key.strip()] = float(value.strip())
                            #if the key str value in dict contains a comma, set its type to list
                            elif ',' in value.strip():
                                current_dict[key.strip()] = value.strip().split(',')
                            #if the key str value in dict contains a colon, set its type to list
                            elif ':' in value.strip():
                                current_dict[key.strip()] = value.strip().split(':')
                            
                            #else set its type to str
                            else:
                                current_dict[key.strip()] = str(value.strip()) #.split(maxsplit=-1)
 
        return namelist

class get_general_info:
    def __init__(self,item,sim_nml,ref_nml,main_nl,sim_source,ref_source,metric_vars):
        self.name = self.__class__.__name__  
        self.minimum_lenghth         = (main_nl['general']['min_year'])
        self.max_lat                 =  main_nl['general']['max_lat']
        self.min_lat                 =  main_nl['general']['min_lat']
        self.max_lon                 =  main_nl['general']['max_lon']
        self.min_lon                 =  main_nl['general']['min_lon']
        self.syear                   =  main_nl['general']['syear']
        self.eyear                   =  main_nl['general']['eyear']
        self.compare_tres            =  main_nl['general']['compare_tres'].lower()
        self.compare_gres            =  main_nl['general']['compare_gres']
        self.casename                =  main_nl['general']['casename']
        self.casedir                 =  os.path.join(main_nl['general']['casedir'], main_nl['general']['casename'])+f'/{item}/{sim_source}___{ref_source}/'
        self.evaluation_only         =  main_nl['general']['evaluation_only']
        self.num_cores               =  main_nl['general']['num_cores']
        self.metrics                 =  metric_vars
        self.compare_tzone           =  main_nl['general']['compare_tzone']

        #for reference data
        self.ref_source              =  ref_source
        self.ref_varname             =  ref_nml[f'{item}'][f'{ref_source}_varname']
        self.ref_data_type           =  ref_nml[f'{item}'][f'{ref_source}_data_type']
        self.ref_data_groupby        =  ref_nml[f'{item}'][f'{ref_source}_data_groupby']
        self.ref_dir                 =  ref_nml[f'{item}'][f'{ref_source}_dir']
        if self.ref_data_type == 'stn':
            self.ref_fulllist            =  ref_nml[f'{item}'][f'{ref_source}_fulllist']

        self.ref_tim_res                 =  ref_nml[f'{item}'][f'{ref_source}_tim_res'].lower()
        self.ref_geo_res                 =  ref_nml[f'{item}'][f'{ref_source}_geo_res']
        self.ref_suffix                  =  ref_nml[f'{item}'][f'{ref_source}_suffix']
        self.ref_prefix                  =  ref_nml[f'{item}'][f'{ref_source}_prefix']
        self.ref_syear                   =  ref_nml[f'{item}'][f'{ref_source}_syear']
        self.ref_eyear                   =  ref_nml[f'{item}'][f'{ref_source}_eyear']

        #for simulation data
        self.sim_model               =  sim_nml[f'{item}'][f'{sim_source}_model']
        self.sim_source              =  sim_source
        self.sim_varname             =  sim_nml[f'{item}'][f'{sim_source}_varname']
        self.sim_data_type           =  sim_nml[f'{item}'][f'{sim_source}_data_type']
        self.sim_data_groupby        =  sim_nml[f'{item}'][f'{sim_source}_data_groupby']
        self.sim_dir                 =  sim_nml[f'{item}'][f'{sim_source}_dir']
        if self.sim_data_type == 'stn':
            self.sim_fulllist        =  sim_nml[f'{item}'][f'{sim_source}_fulllist']
        self.sim_tim_res                 =  sim_nml[f'{item}'][f'{sim_source}_tim_res'].lower()
        self.sim_geo_res                 =  sim_nml[f'{item}'][f'{sim_source}_geo_res']
        self.sim_suffix                  =  sim_nml[f'{item}'][f'{sim_source}_suffix']
        self.sim_prefix                  =  sim_nml[f'{item}'][f'{sim_source}_prefix']
        self.sim_syear                   =  sim_nml[f'{item}'][f'{sim_source}_syear']
        self.sim_eyear                   =  sim_nml[f'{item}'][f'{sim_source}_eyear']
        if self.evaluation_only:
            pass
        else:
            shutil.rmtree(f'{self.casedir}', ignore_errors=True)
            os.makedirs(f'{self.casedir}', exist_ok=True)
        
            #remove tmp directory if exist
            shutil.rmtree(f'{self.casedir}/tmp/sim',ignore_errors=True)
            shutil.rmtree(f'{self.casedir}/tmp/ref',ignore_errors=True)
            #creat tmp directory
            os.makedirs(f'{self.casedir}/tmp/sim', exist_ok=True)
            os.makedirs(f'{self.casedir}/tmp/ref', exist_ok=True)
            os.makedirs(f'{self.casedir}/scratch', exist_ok=True)

        if self.sim_data_type == 'stn' and self.ref_data_type == 'stn':
            self.sim_stn_list=pd.read_csv(self.sim_fulllist,header=0)
            self.ref_stn_list=pd.read_csv(self.ref_fulllist,header=0)
            # check whether all the stations in the sim_stn_list are in the ref_stn_list. if not, warn the user and exit
            if len(self.sim_stn_list[~self.sim_stn_list['ID'].isin(self.ref_stn_list['ID'])])>0:
                print(f"Warning: the following stations are not in the ref_stn_list: {self.sim_stn_list[~self.sim_stn_list['ID'].isin(self.ref_stn_list['ID'])]['ID'].values}")
                sys.exit(1)
            # change the key name SYEAR, EYEAR, Dir in self.sim_stn_list to sim_syear, sim_eyear, sim_dir
            self.sim_stn_list.rename(columns={'SYEAR':'sim_syear','EYEAR':'sim_eyear','DIR':'sim_dir','LON':'sim_lon','LAT':'sim_lat'}, inplace=True)
            # change the key name SYEAR, EYEAR, Dir in self.ref_stn_list to ref_syear, ref_eyear, ref_dir
            self.ref_stn_list.rename(columns={'SYEAR':'ref_syear','EYEAR':'ref_eyear','DIR':'ref_dir','LON':'ref_lon','LAT':'ref_lat'}, inplace=True)
            # find the common stations in the sim_stn_list and ref_stn_list and save them to a new dataframe
            self.stn_list=pd.merge(self.sim_stn_list,self.ref_stn_list,how='inner',on='ID')
            self.stn_list['use_syear']=[-9999] * len(self.stn_list['ref_lat'])   
            self.stn_list['use_eyear']=[-9999] * len(self.stn_list['ref_lat'])
            self.stn_list['Flag']     =[False] * len(self.stn_list['ref_lat'])
            #filter the stations based on the minimum_lenghth, min_lat, max_lat, min_lon, max_lon
            for i in range(len(self.stn_list['ref_lat'])):
                self.stn_list['use_syear'].values[i]=max(self.stn_list['ref_syear'].values[i],self.stn_list['sim_syear'].values[i],self.syear)
                self.stn_list['use_eyear'].values[i]=min(self.stn_list['ref_eyear'].values[i],self.stn_list['sim_eyear'].values[i],self.eyear)
                if ((self.stn_list['use_eyear'].values[i]-self.stn_list['use_syear'].values[i]>=self.minimum_lenghth) &\
                                (self.stn_list['ref_lon'].values[i]>=self.min_lon) &\
                                (self.stn_list['ref_lon'].values[i]<=self.max_lon) &\
                                (self.stn_list['ref_lat'].values[i]>=self.min_lat) &\
                                (self.stn_list['ref_lat'].values[i]<=self.max_lat) 
                                ): 
                    self.stn_list['Flag'].values[i]=True

            ind =  self.stn_list[self.stn_list['Flag']==True].index
            data_select = self.stn_list.loc[ind]
            #save the common stations to a new file
            data_select.to_csv(f"{self.casedir}/stn_list.txt",index=False)
        elif self.sim_data_type == 'stn' and self.ref_data_type != 'stn':
            self.stn_list=pd.read_csv(self.sim_fulllist,header=0)
            # change the key name SYEAR, EYEAR, Dir in self.sim_stn_list to sim_syear, sim_eyear, sim_dir
            self.stn_list.rename(columns={'SYEAR':'sim_syear','EYEAR':'sim_eyear','DIR':'sim_dir','LON':'sim_lon','LAT':'sim_lat'}, inplace=True)
            self.stn_list['use_syear']=[-9999] * len(self.stn_list['ID'])   
            self.stn_list['use_eyear']=[-9999] * len(self.stn_list['ID'])
            self.stn_list['Flag']     =[False] * len(self.stn_list['ID'])
            #filter the stations based on the minimum_lenghth, min_lat, max_lat, min_lon, max_lon
            for i in range(len(self.stn_list['sim_lat'])):
                self.stn_list['use_syear'].values[i]=max(self.ref_syear,self.stn_list['sim_syear'].values[i],self.syear)
                self.stn_list['use_eyear'].values[i]=min(self.ref_eyear,self.stn_list['sim_eyear'].values[i],self.eyear)
                if ((self.stn_list['use_eyear'].values[i]-self.stn_list['use_syear'].values[i]>=self.minimum_lenghth) &\
                                (self.stn_list['sim_lon'].values[i]>=self.min_lon) &\
                                (self.stn_list['sim_lon'].values[i]<=self.max_lon) &\
                                (self.stn_list['sim_lat'].values[i]>=self.min_lat) &\
                                (self.stn_list['sim_lat'].values[i]<=self.max_lat) 
                                ): 
                    self.stn_list['Flag'].values[i]=True

            ind =  self.stn_list[self.stn_list['Flag']==True].index
            data_select = self.stn_list.loc[ind]
            #save the common stations to a new file
            data_select.to_csv(f"{self.casedir}/stn_list.txt",index=False)
        elif self.sim_data_type != 'stn' and self.ref_data_type == 'stn':
            if item == 'Streamflow':
                self.ref_dir             =  ref_nml[f'{item}'][f'{ref_source}_dir']
                self.max_uparea          =  ref_nml[f'{item}'][f'{ref_source}_max_uparea']
                self.min_uparea          =  ref_nml[f'{item}'][f'{ref_source}_min_uparea'] 
                if self.ref_source.lower() == 'grdc':
                    if ((self.compare_tres.lower()=="hour")):
                        print('compare_res="Hour", the compare_res should be "Day","Month" or longer ')
                        sys.exit(1)
                    self.ref_fulllist            =  f"{self.ref_dir}/list/GRDC_alloc_{self.sim_geo_res}Deg.txt"
                    station_list                 =  pd.read_csv(f"{self.ref_fulllist}",delimiter=r"\s+",header=0)
                    station_list['Flag']         =  [False] * len(station_list['lon']) #[False for i in range(len(station_list['lon']))] #[False] * len(station_list['lon'])  #(station_list['lon']*0 -9999)*False
                    station_list['use_syear']    =  [-9999] * len(station_list['lon'])  #int(station_list['lon']*0 -9999)
                    station_list['use_eyear']    =  [-9999] * len(station_list['lon'])
                    station_list['obs_syear']    =  [-9999] * len(station_list['lon']) #must be integer
                    station_list['obs_eyear']    =  [-9999] * len(station_list['lon']) #must be integer
                    if (self.compare_tres.lower() == 'month'):
                        for i in range(len(station_list['ID'])):
                            if(os.path.exists('%s/GRDC_Month/%s_Q_Month.nc'%(self.ref_dir,station_list['ID'][i]))):
                                with xr.open_dataset('%s/GRDC_Month/%s_Q_Month.nc'%(self.ref_dir,station_list['ID'][i])) as df:
                                    station_list['obs_syear'].values[i]=df["time.year"].values[0]
                                    station_list['obs_eyear'].values[i]=df["time.year"].values[-1]
                                    station_list['use_syear'].values[i]=max(station_list['obs_syear'].values[i],self.sim_syear,self.syear)
                                    station_list['use_eyear'].values[i]=min(station_list['obs_eyear'].values[i],self.sim_eyear,self.eyear)
                                    if ((station_list['use_eyear'].values[i]-station_list['use_syear'].values[i]>=self.minimum_lenghth) &\
                                            (station_list['lon'].values[i]>=self.min_lon) &\
                                            (station_list['lon'].values[i]<=self.max_lon) &\
                                            (station_list['lat'].values[i]>=self.min_lat) &\
                                            (station_list['lat'].values[i]<=self.max_lat) &\
                                            (station_list['area1'].values[i]>=self.min_uparea) &\
                                            (station_list['area1'].values[i]<=self.max_uparea) &\
                                            (station_list['ix2'].values[i] == -9999) 
                                            ): 
                                        station_list['Flag'].values[i]=True
                                        print(f"Station ID : {station_list['ID'].values[i]} is selected")
                    elif (self.compare_tres.lower() == 'day'):
                        for i in range(len(station_list['ID'])):
                            if(os.path.exists('%s/GRDC_Day/%s_Q_Day.Cmd.nc'%(self.ref_dir,station_list['ID'][i]))):
                                with xr.open_dataset('%s/GRDC_Day/%s_Q_Day.Cmd.nc'%(self.ref_dir,station_list['ID'][i])) as df:
                                    station_list['obs_syear'].values[i]=df["time.year"].values[0]
                                    station_list['obs_eyear'].values[i]=df["time.year"].values[-1]
                                    station_list['use_syear'].values[i]=max(station_list['obs_syear'].values[i],self.sim_syear,self.syear)
                                    station_list['use_eyear'].values[i]=min(station_list['obs_eyear'].values[i],self.sim_eyear,self.eyear)
                                    if ((station_list['use_eyear'].values[i]-station_list['use_syear'].values[i]>=self.minimum_lenghth) &\
                                            (station_list['lon'].values[i]>=self.min_lon) &\
                                            (station_list['lon'].values[i]<=self.max_lon) &\
                                            (station_list['lat'].values[i]>=self.min_lat) &\
                                            (station_list['lat'].values[i]<=self.max_lat) &\
                                            (station_list['area1'].values[i]>=self.min_uparea) &\
                                            (station_list['area1'].values[i]<=self.max_uparea) &\
                                            (station_list['ix2'].values[i] == -9999) 
                                            ): 
                                        station_list['Flag'].values[i]=True
                                        print(f"Station ID : {station_list['ID'].values[i]} is selected")
                    ind = station_list[station_list['Flag']==True].index
                    data_select = station_list.loc[ind]
                    if self.sim_geo_res==0.25:
                        lat0=np.arange(89.875,-90,-0.25)
                        lon0=np.arange(-179.875,180,0.25)
                    elif self.sim_geo_res==0.0167:#01min
                        lat0=np.arange(89.9916666666666600,-90,-0.0166666666666667)
                        lon0=np.arange(-179.9916666666666742,180,0.0166666666666667)
                    elif self.sim_geo_res==0.0833:#05min
                        lat0=np.arange(89.9583333333333286,-90,-0.0833333333333333)
                        lon0=np.arange(-179.9583333333333428,180,0.0833333333333333)
                    elif self.sim_geo_res==0.1:#06min
                        lat0=np.arange(89.95,-90,-0.1)
                        lon0=np.arange(-179.95,180,0.1)
                    elif self.sim_geo_res==0.05:#03min
                        lat0=np.arange(89.975,-90,-0.05)
                        lon0=np.arange(-179.975,180,0.05)
                    data_select['lon_cama']=[-9999.] * len(data_select['lon']) 
                    data_select['lat_cama']=[-9999.] * len(data_select['lat']) 
                    for iii in range(len(data_select['ID'])):
                        print(iii,len(data_select['ID']))
                        data_select['lon_cama'].values[iii]=float(lon0[int(data_select['ix1'].values[iii])-1])
                        data_select['lat_cama'].values[iii]=float(lat0[int(data_select['iy1'].values[iii])-1])
                        if abs(data_select['lat_cama'].values[iii]-data_select['lat'].values[iii])>1:
                            print(f"Warning: ID {data_select['ID'][iii]} lat is not match")
                        if abs(data_select['lon_cama'].values[iii]-data_select['lon'].values[iii])>1:
                            print(f"Warning: ID {data_select['ID'].values[iii]} lon is not match")

                    # print(data_select)
                    print(f"In total: {len(data_select['ID'])} stations are selected")
                    #if len(data_select['ID'])==0, exit
                    if len(data_select['ID'])==0:
                        print(f"Warning: No stations are selected, please check the station list and the minimum_lenghth, min_lat, max_lat, min_lon, max_lon")
                        sys.exit(1)
                    data_select.to_csv(f"{self.casedir}/stn_list.txt",index=False)
            else:
                self.ref_stn_list=pd.read_csv(self.ref_fulllist,header=0)
                # change the key name SYEAR, EYEAR, Dir in self.ref_stn_list to ref_syear, ref_eyear, ref_dir
                self.ref_stn_list.rename(columns={'SYEAR':'ref_syear','EYEAR':'ref_eyear','DIR':'ref_dir','LON':'ref_lon','LAT':'ref_lat'}, inplace=True)
                # find the common stations in the sim_stn_list and ref_stn_list and save them to a new dataframe
                self.stn_list['use_syear']=[-9999] * len(self.stn_list['ref_lat'])   
                self.stn_list['use_eyear']=[-9999] * len(self.stn_list['ref_lat'])
                self.stn_list['Flag']     =[False] * len(self.stn_list['ref_lat'])
                #filter the stations based on the minimum_lenghth, min_lat, max_lat, min_lon, max_lon
                for i in range(len(self.stn_list['ref_lat'])):
                    self.stn_list['use_syear'].values[i]=max(self.stn_list['ref_syear'].values[i],self.stn_list['sim_syear'].values[i],self.syear)
                    self.stn_list['use_eyear'].values[i]=min(self.stn_list['ref_eyear'].values[i],self.stn_list['sim_eyear'].values[i],self.eyear)
                    if ((self.stn_list['use_eyear'].values[i]-self.stn_list['use_syear'].values[i]>=self.minimum_lenghth) &\
                            (self.stn_list['ref_lon'].values[i]>=self.min_lon) &\
                            (self.stn_list['ref_lon'].values[i]<=self.max_lon) &\
                            (self.stn_list['ref_lat'].values[i]>=self.min_lat) &\
                            (self.stn_list['ref_lat'].values[i]<=self.max_lat) 
                        ): 
                        self.stn_list['Flag'].values[i]=True

                ind =  self.stn_list[self.stn_list['Flag']==True].index
                data_select = self.stn_list.loc[ind]
                #save the common stations to a new file
                data_select.to_csv(f"{self.casedir}/stn_list.txt",index=False)


        elif self.sim_data_type != 'stn' and self.ref_data_type != 'stn':
            self.use_syear               =  max(self.syear, self.sim_syear, self.ref_syear)
            self.use_eyear               =  min(self.eyear, self.sim_eyear, self.ref_eyear)

        else:
            print("Error: the reference data type and simulation data type are not consistent!")
            sys.exit(1)


        # return self as a dictionary
        self_dict = self.__dict__

    def to_dict(self):

        return self.__dict__
'''
    def split_year(self,casedir,dir,suffix,prefix,tim_res,syear,use_syear,use_eyear):
        # Open the netCDF file
        VarFile = os.path.join(dir, f'{suffix}{prefix}.nc')
        ds = xr.open_dataset(VarFile)
        num=len(ds['time'])    
        #this is not good, need to be changed
        if (tim_res.lower()=="hour"):
            freq="H"
        elif (tim_res.lower()=="day"):
            freq="D"
        elif (tim_res.lower()=="month"):
            freq="M"
        elif (tim_res.lower()=="year"):
            freq="Y"
        else:
            print('sim_tim_res error')
            sys.exit(1)
        if any(ds['time'].dt.dayofyear) == 366:
            ds['time'] = pd.date_range(f"{syear}-01-01", freq=f'{freq}', periods=num,calendar='standard')
        else:
            ds['time'] = xr.cftime_range(start=f"{syear}-01-01", freq=f'{freq}', periods=num, calendar="noleap") 

        # Split the data into yearly files
        for year in range(use_syear, use_eyear+1):
            ds_year = ds.sel(time=slice(f'{year}-01-01T00:00:00',f'{year}-12-31T23:59:59'))
            # Remove all attributes
            ds_year.attrs = {}
            ds_year.to_netcdf(os.path.join(casedir,'scratch',f'{suffix}{year}{prefix}.nc'))
            ds.close()




'''
