# -*- coding: utf-8 -*-
import os
import pandas as pd
import xarray as xr
import numpy as np
import sys
import shutil 
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'

# Check the platform
if not sys.platform.startswith('win'):
    import xesmf as xe

class Makefiles_parallel:
    def __init__(self, info):
        self.name    = 'Makefile'
        self.version = '0.1'
        self.release = '0.1'
        self.date    = 'Mar 2023'
        self.author  = "Zhongwang Wei / zhongwang007@gmail.com"
        #copy all the dict in info to self
        self.__dict__.update(info)
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]
        
    def make_combine_parallel(self,dy,TimRes,ii):
        k  = getattr(self, f'{dy}_dir')
        k1 = getattr(self, f'{dy}_suffix')
        k2 = getattr(self, f'{dy}_prefix')
        k3 = getattr(self, f'{dy}_varname')
        VarFiles=f'{k}/{k1}{ii}*{k2}.nc'
        print(VarFiles)
        with xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time",decode_times=False, parallel=False,preprocess=lambda dfx: dfx[k3[0]].astype('float32')) as dfx:
            #if (self.sim_model=='CoLM') then rename lon_cama to lon and lat_cama to lat
            if (self.sim_model=='CoLM'):
                dfx['lon']=dfx['lon_cama']
                dfx['lat']=dfx['lat_cama']    
            num=len(dfx['time'])
            freq_map = {'hour': 'H', 'day': 'D', 'month': 'M'}
            if TimRes.lower() in freq_map:
                dfx['time'] = pd.date_range(f"{ii}-01-01", freq=freq_map[TimRes.lower()], periods=num)
            else:
                sys.exit(1)

            compare_tres_map = {'month': '1M', 'day': '1D', 'hour': '1H'}
            if self.compare_tres.lower() in compare_tres_map:
                dfx = dfx.resample(time=compare_tres_map[self.compare_tres.lower()]).mean()
            else:
                sys.exit(1)

            dfx=dfx.sel(time=slice(f'{ii}-01-01',f'{ii}-12-31'))
            mask_lon = (dfx.lon >= self.min_lon) & (dfx.lon <= self.max_lon)
            mask_lat = (dfx.lat >= self.min_lat) & (dfx.lat <= self.max_lat)
            cropped_ds = dfx.where(mask_lon & mask_lat, drop=True)
            delayed_obj=cropped_ds.to_netcdf(f'{self.casedir}/tmp/{dy}/'+f'{dy}_{ii}.nc', compute=False)
            with ProgressBar():
                delayed_obj.compute()
            print(f'Year {ii}: Files Combined')
            dfx.close()
        del  cropped_ds,mask_lon,mask_lat,delayed_obj,num
        del VarFiles

    def make_geo_sim_parallel(self,ii):
        VarFiles=(f'{self.sim_dir}/{self.sim_suffix}{ii}{self.sim_prefix}.nc')
        simx0= xr.open_dataset(VarFiles)
        if self.sim_model=='CoLM':
            if 'lon' not in simx0:
                simx0['lon']=simx0['lon_cama']
            if 'lat' not in simx0:
                simx0['lat']=simx0['lat_cama']
        else:
            if 'lon' not in simx0:
                simx0['lon']=simx0['longitude']
            if 'lat' not in simx0:
                simx0['lat']=simx0['latitude']
        #need check here
        simx0['lon'] = (simx0['lon'] + 180) % 360 - 180.0
        lon_new = xr.DataArray(
                data=np.arange(self.min_lon+self.compare_gres/2, self.max_lon, self.compare_gres),
                dims=('lon',),
                coords={'lon': np.arange(self.min_lon+self.compare_gres/2, self.max_lon, self.compare_gres)},
                attrs={'units': 'degrees_east', 'long_name': 'longitude'}
                )
        lat_new = xr.DataArray(
                data=np.arange(self.min_lat+self.compare_gres/2, self.max_lat, self.compare_gres),
                dims=('lat',),
                coords={'lat': np.arange(self.min_lat+self.compare_gres/2, self.max_lat, self.compare_gres)},
                attrs={'units': 'degrees_north', 'long_name': 'latitude'}
                )
        new_grid = xr.Dataset({'lon': lon_new, 'lat': lat_new})
            
        if not sys.platform.startswith('win'):
            # Get the data variable
            #refx = refx[f'{key}'] #['fldfrc']
            # Create the regridder
            # Define the path to the weights file
            if (self.min_lon==-180 and self.max_lon==180 and self.min_lat==-90 and self.max_lat==90):
                regridder = xe.Regridder(simx0, new_grid, 'bilinear', periodic=True)
            else:
                regridder = xe.Regridder(simx0, new_grid, 'bilinear', periodic=False)
                    # Perform the remapping
            simx = regridder(simx0)
        else:
            simx = simx0.interp(coords=new_grid.coords) 
        
        num=len(simx['time'])
        freq_map = {'hour': 'H', 'day': 'D', 'month': 'M','year' : 'Y'}
        if self.sim_tim_res.lower() in freq_map:
            simx['time'] = pd.date_range(f"{ii}-01-01", freq=freq_map[self.sim_tim_res.lower()], periods=num)
        else:
            sys.exit(1)     


        if (self.compare_tres.lower() =="month"):
            simx=simx.resample(time='1M').mean() 
        elif (self.compare_tres.lower() =="day"):
            simx=simx.resample(time='1D').mean() 
        elif (self.compare_tres.lower() =="hour"):
            simx=simx.resample(time='1H').mean() 
        elif (self.compare_tres.lower() =="year"):
            simx=simx.resample(time='1Y').mean()
        else:
            print('check self.compare_tres')
            sys.exit(1)  

        simx=simx.sel(time=slice(f'{ii}-01-01T00:00:00',f'{ii}-12-31T23:59:59'))
        simxvar=simx[f'{self.sim_varname[0]}']
        # Save the output dataset to a netcdf file
        out_file = f'{self.casedir}/tmp/sim/'+f'sim_{self.sim_varname[0]}_remap_{ii}.nc'
        simxvar.to_netcdf(out_file) 
        del simxvar,simx0,simx,regridder,VarFiles
        print(f"Done with Year {ii}")

    def make_geo_ref_parallel(self,ii):
        if (self.ref_source=='GIEMS_v2_2020') :
            VarFiles = os.path.join(self.ref_dir, f'GIEMS_v2_2020_{self.ref_varname}_15min_{ii}.nc')
        elif (self.ref_source=='GLEAM'):
            VarFiles = os.path.join(self.ref_dir, 'daily', f'{self.ref_varname}_{ii}_GLEAM_v3.7a.nc')
        elif (self.ref_source=='GLDAS'):
            print('not ready yet')
            sys.exit(1)
        elif ((self.ref_source=='ERA5-Land')and(self.ref_varname=='ro')):
            VarFiles = os.path.join(self.ref_dir, f'ERA5LAND_runoff_{ii}.nc4') ###check here_2018_GLEAM_v3.7a.nc
        elif ((self.ref_source=='Yuan_etal')and(self.ref_varname=='lai')):
            VarFiles = os.path.join(self.ref_dir, f'lai_8-day_30s_{ii}.nc4') ###check here_2018_GLEAM_v3.7a.nc
        else:
            VarFiles = os.path.join(self.ref_dir, f'{self.ref_suffix}{ii}{self.ref_prefix}.nc')
        refx0 = xr.open_dataset(VarFiles)
            

        if 'lon' not in refx0:
            refx0['lon']=refx0['longitude']
        if 'lat' not in refx0:
            refx0['lat']=refx0['latitude']
        #need check here
        refx0['lon'] = (refx0['lon'] + 180) % 360 - 180.0
        #refx0['lon'] = xr.where(refx0.lon > 180, refx0.lon - 360, refx0.lon)

        lon_new = xr.DataArray(
                data=np.arange(self.min_lon+self.compare_gres/2, self.max_lon, self.compare_gres),
                dims=('lon',),
                coords={'lon': np.arange(self.min_lon+self.compare_gres/2, self.max_lon, self.compare_gres)},
                attrs={'units': 'degrees_east', 'long_name': 'longitude'}
                )
        lat_new = xr.DataArray(
                data=np.arange(self.min_lat+self.compare_gres/2, self.max_lat, self.compare_gres),
                dims=('lat',),
                coords={'lat': np.arange(self.min_lat+self.compare_gres/2, self.max_lat, self.compare_gres)},
                attrs={'units': 'degrees_north', 'long_name': 'latitude'}
                )
        new_grid = xr.Dataset({'lon': lon_new, 'lat': lat_new})
 
        if not sys.platform.startswith('win'):
            if (self.min_lon==-180 and self.max_lon==180 and self.min_lat==-90 and self.max_lat==90):
                regridder = xe.Regridder(refx0, new_grid, 'bilinear', periodic=True)
            else:
                regridder = xe.Regridder(refx0, new_grid, 'bilinear', periodic=False)
                # Perform the remapping
            refx = regridder(refx0)
        else:
            refx = refx[f'{self.ref_varname}'] 
            refx = refx.interp(coords=new_grid.coords) 

        #need to improve here
        num=len(refx['time'])
        ref_tim_res_map = {'month': '1M', 'day': '1D', 'hour': '1H', 'year':'1Y'}
        if self.ref_tim_res.lower() in ref_tim_res_map:
             refx['time'] = pd.date_range(f"{ii}-01-01", freq=ref_tim_res_map[self.ref_tim_res.lower()], periods=num)
        else:
            sys.exit(1)


        
        # Resample based on compare_tres
        compare_tres_map = {'month': '1M', 'day': '1D', 'hour': '1H', 'year':'1Y'}
        if self.compare_tres.lower() in compare_tres_map:
            refx = refx.resample(time=compare_tres_map[self.compare_tres.lower()]).mean()
        else:
            sys.exit(1)

        refx=refx.sel(time=slice(f'{ii}-01-01T00:00:00',f'{ii}-12-31T23:59:59'))

        # Save the output dataset to a netcdf file
        out_file = f'{self.casedir}/tmp/ref/'+f'ref_{self.ref_varname[0]}_remap_{ii}.nc'
        refx.to_netcdf(out_file)
        del refx0,refx,regridder,VarFiles
        print(f"Done with Year {ii}") 

    def make_stn_ref_parallel(self,station_list,i):
        if (self.ref_source.lower()=='grdc'):
            print(f"deal with reference station: {station_list['ID'][i]}")
            if (self.compare_tres.lower() == 'month'):
                stn=xr.open_dataset('%s/GRDC_Month/%s_Q_Month.nc'%(self.ref_dir,station_list['ID'][i]))
            elif (self.compare_tres.lower() == 'day'):
                stn=xr.open_dataset('%s/GRDC_Day/%s_Q_Day.Cmd.nc'%(self.ref_dir,station_list['ID'][i]))
        elif (self.ref_source=='PLUMBER2'):
            print(f"deal with reference station: {station_list['ID'][i]}")
            try:
                print(f"{station_list['ref_dir'][i]}"+f"{station_list['ID'][i]}"+"*.nc")
                stn=xr.open_mfdataset(f"{station_list['ref_dir'][i]}"+f"{station_list['ID'][i]}"+"*.nc",
                                    combine='nested',concat_dim="time",decode_times=True,chunks={'time': 30},parallel=False, 
                                    preprocess=lambda refx: refx[self.ref_varname].astype('float32')).squeeze()
            except:
                kk=self.ref_varname[0:-4]
                stn=xr.open_mfdataset(f"{station_list['ref_dir'][i]}"+f"{station_list['ID'][i]}"+"*.nc",
                                    combine='nested',concat_dim="time",decode_times=True,chunks={'time': 30},parallel=False, 
                                    preprocess=lambda refx: refx[kk].astype('float32')).squeeze()
                # rename kk to self.ref_varname
                stn=stn.rename({kk:self.ref_varname})
                del kk
            #drop the x and y coordinate for the PLUMBER2 data
            #df[self.ref_varname]=df[self.ref_varname].drop(['x','y'])
        elif (self.ref_source=='ismn'):
            stn=f'{self.casedir}/scratch/'+station_list['ID'][i]+'.nc'
        elif (self.ref_source=='GLEAM_hybird'):
            stn=os.path.join(self.Obs_Dir,station_list['ID'][i])+'.nc'
        elif (self.ref_source=='GLEAM_hybird_PLUMBER2'):
            stn=f'{self.casedir}/scratch/'+station_list['ID'][i]+'.nc'
        elif (self.ref_source=='Yuan2022'):
            stn=f'{self.casedir}/scratch/'+station_list['ID'][i]+'.nc'
        elif (self.ref_source=='HydroWeb_2.0'):
            stn=f'{self.casedir}/scratch/'+station_list['ID'][i]+'.nc'
        elif (self.ref_source=='ResOpsUS'):
            stn=f'{self.casedir}/scratch/'+station_list['ID'][i]+'.nc'
        startx=int(station_list['use_syear'].values[i])
        endx  =int(station_list['use_eyear'].values[i])
        df = stn #xr.decode_cf(stn[self.ref_varname])

        dfx1=df.sel(time=slice(f'{startx}-01-01',f'{endx}-12-31')) 
        if (self.compare_tres.lower() == 'month'):
            dfx2=dfx1.resample(time='1M').mean()
            time_index = pd.date_range(start=f'{startx}-01-01', end=f'{endx}-12-31', freq='M')
        elif (self.compare_tres.lower() == 'day'):
            dfx2=dfx1.resample(time='1D').mean()
            time_index = pd.date_range(start=f'{startx}-01-01', end=f'{endx}-12-31', freq='D')
        elif (self.compare_tres.lower() == 'hour'):
            dfx2=dfx1.resample(time='1H').mean()
            time_index = pd.date_range(start=f'{startx}-01-01', end=f'{endx}-12-31', freq='H')       
        elif (self.compare_tres.lower() == 'year'):
            dfx2=dfx1.resample(time='1Y').mean()
            time_index = pd.date_range(start=f'{startx}-01-01', end=f'{endx}-12-31', freq='Y')   
        else:
            print('check self.compare_tres')
            sys.exit(1)
        # Create empty xarray dataset with time index
        ds = xr.Dataset({'data': (['time'], np.nan*np.ones(len(time_index)))},coords={'time': time_index})

        # Reindex original dataset to match new time index
        orig_ds_reindexed = dfx2.reindex(time=ds.time)
        # Merge original and new datasets
        merged_ds = xr.merge([ds, orig_ds_reindexed]).drop_vars('data')
        #keep only time coordinate/dimension
        try:
            merged_ds=merged_ds.drop(['x','y'])
        except:
            pass
        # Save the output dataset to a netcdf file, if file exist, then delete it
        try:
            os.remove(f'{self.casedir}/tmp/ref/'+f"ref_{station_list['ID'][i]}"+f"_{station_list['use_syear'][i]}"+f"_{station_list['use_eyear'][i]}.nc")
        except:
            pass
        merged_ds.to_netcdf(f'{self.casedir}/tmp/ref/'+f"ref_{station_list['ID'][i]}"+f"_{station_list['use_syear'][i]}"+f"_{station_list['use_eyear'][i]}.nc")
        del startx,endx,dfx1,dfx2,ds,orig_ds_reindexed,merged_ds,time_index,df,stn

    def make_stn_sim_parallel(self,station_list,i):
        print(f"deal with simulation station: {station_list['ID'][i]}")
        VarFiles = os.path.join(f"{station_list['sim_dir'][i]}", f"{station_list['ID'][i]}*.nc")
        print(VarFiles)
        df= xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time",parallel=False, decode_times=True,chunks={'time': 300},
                                   preprocess=lambda simx: simx[self.sim_varname].astype('float32')).squeeze()
        startx=int(station_list['use_syear'].values[i])
        endx  =int(station_list['use_eyear'].values[i])
        ## Decode the time units and reduce the dimension
        dfx = df # xr.decode_cf(df)
        dfx1=dfx.sel(time=slice(f'{startx}-01-01',f'{endx}-12-31')) 


        # Resample based on compare_tres
        compare_tres_map = {'month': '1M', 'day': '1D', 'hour': '1H', 'year':'1Y'}
        if self.compare_tres.lower() in compare_tres_map:
            dfx2 = dfx1.resample(time=compare_tres_map[self.compare_tres.lower()]).mean()
            time_index = pd.date_range(start=f'{startx}-01-01', end=f'{endx}-12-31', freq=compare_tres_map[self.compare_tres.lower()])
        else:
            sys.exit(1)

        # Create empty xarray dataset with time index
        ds = xr.Dataset({'data': (['time'], np.nan*np.ones(len(time_index)))},coords={'time': time_index})
        # Reindex original dataset to match new time index
        orig_ds_reindexed = dfx2.reindex(time=ds.time)
        # Merge original and new datasets
        merged_ds = xr.merge([ds, orig_ds_reindexed]).drop_vars('data')
        try:
            os.remove(f'{self.casedir}/tmp/sim/'+f"sim_{station_list['ID'][i]}"+f"_{station_list['use_syear'][i]}"+f"_{station_list['use_eyear'][i]}.nc")
        except:
            pass
        merged_ds.to_netcdf(f'{self.casedir}/tmp/sim/'+f"sim_{station_list['ID'][i]}"+f"_{station_list['use_syear'][i]}"+f"_{station_list['use_eyear'][i]}.nc")
        del startx,endx,dfx,dfx1,dfx2,ds,orig_ds_reindexed,merged_ds,time_index,df,VarFiles

    def extract_stn_parallel(self,dy,dataset,station_list,ik):
        print(f"deal with station: {station_list['ID'][ik]}")
        startx=int(station_list['use_syear'].values[ik])
        endx  =int(station_list['use_eyear'].values[ik])
            
        if (self.ref_source.lower()=='grdc'):
            if (self.sim_model=='CoLM'):
                dataset1=dataset.sel(lat_cama=[station_list['lat_cama'].values[ik]], lon_cama=[station_list['lon_cama'].values[ik]], method="nearest")
            else:
                dataset1=dataset.sel(lat=[station_list['lat_cama'].values[ik]], lon=[station_list['lon_cama'].values[ik]], method="nearest")
        elif (self.ref_source.lower()=='resopsus'):
            dataset1=dataset.sel(lat=[station_list['lat_cama'].values[ik]], lon=[station_list['lon_cama'].values[ik]], method="nearest")
        else:
            #caution here
            if dy=='ref':
                dataset1=dataset.sel(lat=[station_list['sim_lat'].values[ik]], lon=[station_list['sim_lon'].values[ik]], method="nearest")
            elif dy=='sim':
                if (self.sim_model=='CoLM'):
                    dataset1=dataset.sel(lat_cama=[station_list['ref_lat'].values[ik]], lon_cama=[station_list['ref_lon'].values[ik]], method="nearest")
                else:
                    dataset1=dataset.sel(lat=[station_list['ref_lat'].values[ik]], lon=[station_list['ref_lon'].values[ik]], method="nearest")
            else:
                print('check dy')
                sys.exit(1)
        dataset2=dataset1.sel(time=slice(f'{startx}-01-01T00:00:00',f'{endx}-12-31T23:59:59'))
        dataset2.to_netcdf(f"{self.casedir}/tmp/{dy}/{dy}_{station_list['ID'][ik]}"+f"_{station_list['use_syear'][ik]}"+f"_{station_list['use_eyear'][ik]}.nc")
        del dataset1,dataset2,startx,endx,ik,dataset,station_list

    def Makefiles_parallel(self):
        print("=======================================")
        print("Create directory!")
        print("***************************************")
        print(" ")
        timeout=9999
        #print self contents

        if self.ref_data_type == 'stn' or self.sim_data_type == 'stn':
            shutil.rmtree (f'{self.casedir}/tmp/plt',ignore_errors=True)
            os.makedirs   (f'{self.casedir}/tmp/plt', exist_ok=True)

        print(f"tmp directory: {self.casedir}/tmp has been created!")
        print("=======================================")
        print(" ")
        print(" ")
        num_cores = self.num_cores #os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count(),或者随意设定<=cpu核心数的数值
        if self.ref_data_type != 'stn' and self.sim_data_type != 'stn':
            minyear=self.use_syear
            maxyear=self.use_eyear
            #deal with observation data
            print("=======================================")
            print("deal with reference data")
            print(" ")
            print(" ")
                            
            if self.ref_data_groupby.lower() != 'year':
                print ('The Obs_DataGroupby is not Year-->combine it to Year')
                for ii in range(minyear, maxyear+1):
                    VarFiles = os.path.join(self.ref_dir, f'{self.ref_suffix}{ii}*{self.ref_prefix}.nc')
                    refx0= xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time",decode_times=False,chunks={'time': 300}, parallel=True,
                                   preprocess=lambda refx: refx[self.sim_varname].astype('float32'))
                    refx0.to_netcdf(f'{self.casedir}/scratch/'+f'ref_{self.ref_varname[0]}_{ii}.nc')
                self.ref_dir=self.casedir+'/scratch'
                self.ref_suffix=f'ref_{self.ref_varname[0]}_'
                print("done with combine sim data to year")
                
            Parallel(n_jobs=num_cores, timeout=timeout)(delayed(self.make_geo_ref_parallel)(i) for i in range((minyear),(maxyear)+1))
            VarFiles=(f'{self.casedir}/tmp/ref/'+f'ref_{self.ref_varname[0]}_remap_*.nc')
            with xr.open_mfdataset(VarFiles, combine='nested',parallel=False,concat_dim="time",chunks={'time': 30}) as ds1: #,parallel=True,autoclose=True
                delayed_obj=ds1.to_netcdf(f'{self.casedir}/tmp/ref/ref_{self.ref_varname[0]}.nc', compute=False)
                with ProgressBar():
                    delayed_obj.compute()
            del ds1, delayed_obj,VarFiles 
            for ii in range((minyear),(maxyear)+1):
                os.remove(f'{self.casedir}/tmp/ref/'+f'ref_{self.ref_varname[0]}_remap_{ii}.nc')
            print(" ")
            print(" ")   
            print ('Reference data prepared!')
            print("=======================================")
            print(" ")
            print(" ")   
            
            #deal with simulation data
            print("=======================================")
            print("deal with simulation data")
            print(" ")
            print(" ")
            if self.sim_data_groupby.lower() != 'year':
                print ('The Sim_DataGroupby is not Year-->combine it to Year')
                for ii in range(minyear, maxyear+1):
                    VarFiles = os.path.join(self.sim_dir, f'{self.sim_suffix}{ii}*{self.sim_prefix}.nc')
                    print(VarFiles)
                    simx0= xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time",decode_times=False,chunks={'time': 30}, parallel=True,
                                   preprocess=lambda simx: simx[self.sim_varname].astype('float32'))
                    simx0.to_netcdf(f'{self.casedir}/scratch/'+f'sim_{self.sim_varname[0]}_{ii}.nc')
                self.sim_dir=self.casedir+'/scratch'
                self.sim_suffix=f'sim_{self.sim_varname[0]}_'
                print("done with combine sim data to year")
            Parallel(n_jobs=num_cores, timeout=timeout)(delayed(self.make_geo_sim_parallel)(i) for i in range((minyear),(maxyear)+1))
            VarFiles=(f'{self.casedir}/tmp/sim/'+f'sim_{self.sim_varname[0]}_remap_*.nc')
            with xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time",parallel=False,chunks={'time': 30}) as ds1: #,parallel=True,autoclose=True
                delayed_obj=ds1.to_netcdf(f'{self.casedir}/tmp/sim/sim_{self.sim_varname[0]}.nc', compute=False)
                with ProgressBar():
                    delayed_obj.compute()

            del ds1, delayed_obj, VarFiles
            for ii in range((minyear),(maxyear)+1):
                os.remove(f'{self.casedir}/tmp/sim/'+f'sim_{self.sim_varname[0]}_remap_{ii}.nc')
            print(" ")
            print(" ") 
            print ('simulation data prepared!')
            print("=======================================")
            print(" ")
            print(" ")  

        elif self.ref_data_type != 'stn' and self.sim_data_type == 'stn':
            #deal with observation data
            print("=======================================")
            print("deal with reference data")
            print(" ")
            print(" ") 
                     
            stnlist  =f"{self.casedir}/stn_list.txt"
            station_list = pd.read_csv(stnlist,header=0)
            minyear=min(station_list['use_syear'].values[:])
            maxyear=max(station_list['use_eyear'].values[:])
            Parallel(n_jobs=num_cores)(delayed(self.make_combine_parallel)('ref',self.ref_tim_res,ii) for ii in range((minyear),(maxyear)+1))

            VarFiles=(f'{self.casedir}/tmp/ref/ref_*.nc')
            with xr.open_mfdataset(VarFiles, combine='nested',parallel=False,concat_dim="time") as ds1: #,parallel=True,autoclose=True
                delayed_obj=ds1.to_netcdf(f'{self.casedir}/tmp/ref/ref.nc', compute=False)
                with ProgressBar():
                    delayed_obj.compute()
            ds1.close()

            del ds1, delayed_obj,VarFiles,maxyear,minyear
            shutil.rmtree(f'{self.casedir}/tmp/ref/ref_*.nc',ignore_errors=True)
            with xr.open_dataset(f'{self.casedir}/tmp/ref/ref.nc') as refx:
                Parallel(n_jobs=num_cores)(delayed(self.extract_stn_parallel)('ref',refx,station_list,i) for i in range(len(station_list['ID'])))
            del refx
            print(" ")
            print(" ") 
            print ('Reference data prepared!')
            print("=======================================")
            print(" ")
            print(" ") 

            print("=======================================")
            print("deal with Simulation data")
            print(" ")
            print(" ")
            Parallel(n_jobs=num_cores)(delayed(self.make_stn_sim_parallel)(station_list,i) for i in range(len(station_list['ID'])))
            print(" ")
            print(" ")            
            print ('Simulation data prepared!')
            print("=======================================")
            print(" ")
            print(" ")   

        elif self.ref_data_type == 'stn' and self.sim_data_type != 'stn':
            #deal with observation data
            print("=======================================")
            print("deal with reference data")
            print(" ")
            print(" ")
            #read station information
            stnlist  =f"{self.casedir}/stn_list.txt"
            station_list = pd.read_csv(stnlist,header=0)
            Parallel(n_jobs=num_cores)(delayed(self.make_stn_ref_parallel)(station_list,i) for i in range(len(station_list['ID'])))
            print(" ")
            print(" ")               
            
            print ('Reference data prepared!')
            print("=======================================")
            print(" ")
            print(" ")   
            #deal with simulation data
            print("=======================================")
            print("deal with simulation data")
            print(" ")
            print(" ")
            minyear=min(station_list['use_syear'].values[:])
            maxyear=max(station_list['use_eyear'].values[:])

            Parallel(n_jobs=num_cores)(delayed(self.make_combine_parallel)('sim',self.sim_tim_res,ii) for ii in range((minyear),(maxyear)+1))
            VarFiles=(f'{self.casedir}/tmp/sim/sim_*.nc')
            print(VarFiles)
            with xr.open_mfdataset(VarFiles, combine='nested',concat_dim="time") as ds1: #,parallel=True,autoclose=True
                delayed_obj=ds1.to_netcdf(f'{self.casedir}/tmp/sim/sim.nc', compute=False)
                with ProgressBar():
                    delayed_obj.compute()
            ds1.close()

            del ds1, delayed_obj
            shutil.rmtree(f'{self.casedir}/tmp/sim/sim_*.nc',ignore_errors=True)
            with xr.open_dataset(f'{self.casedir}/tmp/sim/sim.nc') as simx:
                Parallel(n_jobs=num_cores)(delayed(self.extract_stn_parallel)('sim',simx,station_list,i) for i in range(len(station_list['ID'])))
            simx.close()
            del simx

        elif self.ref_data_type == 'stn' and self.sim_data_type == 'stn':
            #deal with observation data
            print("=======================================")
            print("deal with reference data")
            print(" ")
            print(" ")
             #read station information
            stnlist  =f"{self.casedir}/stn_list.txt"
            station_list = pd.read_csv(stnlist,header=0)
            Parallel(n_jobs=num_cores)(delayed(self.make_stn_ref_parallel)(station_list,i) for i in range(len(station_list['ID'])))
            print(" ")
            print(" ")   
            print ('Reference data prepared!')
            print("=======================================")
            print(" ")
            print(" ")   

            #deal with simulation data
            print("=======================================")
            print("deal with simulation data")
            print(" ")
            print(" ")
            #stnlist  =f"{self.casedir}/stn_list.txt"
            #station_list = pd.read_csv(stnlist,header=0)
            Parallel(n_jobs=num_cores)(delayed(self.make_stn_sim_parallel)(station_list,j) for j in range(len(station_list['ID'])))
            print(" ")
            print(" ")   
            print ('simulation data prepared!')
            print("=======================================")
            print(" ")
            print(" ")   
  
        else:
            print('Something worng in data_type setting')









