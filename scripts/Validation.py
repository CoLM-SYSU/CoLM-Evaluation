import numpy as np
import os, sys
import xarray as xr
import shutil 
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import colors
# Check the platform
from metrics import metrics_geo,metrics_stn
from scores import scores_geo,scores_stn

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

class Validation_geo(metrics_geo,scores_geo):
    def __init__(self, info):
        self.name = 'Validation'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)

        os.makedirs(self.casedir+'/output/', exist_ok=True)

        print('Validation processes starting!')
        print("=======================================")
        print(" ")
        print(" ")

    def process_metric(self, key, metric, s, o):
        pb = getattr(self, metric)(s, o)
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=metric)
        pb_da.to_netcdf(f'{self.casedir}/output/{key}_{metric}.nc')

    def process_score(self, key, score, s, o):
        pb = getattr(self, score)(s, o)
        pb_da = xr.DataArray(pb, coords=[o.lat, o.lon], dims=['lat', 'lon'], name=score)
        pb_da.to_netcdf(f'{self.casedir}/output/{key}_{score}.nc')

    def make_validation(self, **kwargs):
        o = xr.open_dataset(f'{self.casedir}/tmp/ref/' + f'ref_{self.ref_varname}.nc')[f'{self.ref_varname}'] 
        s = xr.open_dataset(f'{self.casedir}/tmp/sim/' + f'sim_{self.sim_varname}.nc')[f'{self.sim_varname}'] 
        print(o, s)
        print(self.ref_varname)
        if self.ref_varname in ['E', 'Et', 'Ei', 'Es']:
            o = o / 86400
        if self.ref_varname == 'SMsurf':
            s = s[:, 0, :, :].squeeze()

        if self.ref_varname == 'SMroot':
            s = s[:, 1, :, :].squeeze()

        s['time'] = o['time']

        mask1 = np.isnan(s) | np.isnan(o)
        s.values[mask1] = np.nan
        o.values[mask1] = np.nan
        
        for metric in self.metrics:
            print(metric)
            if hasattr(self, metric):
                self.process_metric(self.ref_varname, metric, s, o)
            else:
                print(metric)
                print('No such metric')
                sys.exit(1)

        for score in self.scores:
            print(score)
            if hasattr(self, score):
                self.process_score(self.ref_varname, score, s, o)
            else:
                print(score)
                print('No such score')
                sys.exit(1)

        print("=======================================")
        print(" ")
        print(" ")

        return

    
    def make_plot_index(self):
        from matplotlib import colors
        key=self.ref_varname
        for metric in self.metrics:
            print(f'plotting metric: {metric}')
            if metric in ['bias', 'mae', 'ubRMSE', 'apb', 'RMSE', 'L','pc_bias','apb']:
                vmin = -100.0
                vmax = 100.0
            elif metric in ['KGE', 'NSE', 'correlation']:
                vmin = -1
                vmax = 1
            elif metric in ['correlation_R2', 'index_agreement']:
                vmin = 0
                vmax = 1
            else:
                vmin = -1
                vmax = 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            self.plot_map(cmap, norm, key, bnd,metric)
        print("=======================================")
        for score in self.scores:
            print(f'plotting score: {score}')
            if score in ['KGESS']:
                vmin = -1
                vmax = 1
            elif score in ['nBiasScore','nRMSEScore']:
                vmin = 0
                vmax = 1
            else:
                vmin = -1
                vmax = 1
            bnd = np.linspace(vmin, vmax, 11)
            cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
            cmap = colors.ListedColormap(cpool)
            norm = colors.BoundaryNorm(bnd, cmap.N)
            self.plot_map(cmap, norm, key, bnd,score)
        print("=======================================")

    def plot_map(self, colormap, normalize, key, levels, xitem, **kwargs):
        # Plot settings
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        import xarray as xr
        from mpl_toolkits.basemap import Basemap
        from matplotlib import rcParams
        font = {'family': 'DejaVu Sans'}
        matplotlib.rc('font', **font)

        params = {'backend': 'ps',
                  'axes.labelsize': 10,
                  'grid.linewidth': 0.2,
                  'font.size': 12,
                  'legend.fontsize': 12,
                  'legend.frameon': False,
                  'xtick.labelsize': 12,
                  'xtick.direction': 'out',
                  'ytick.labelsize': 12,
                  'ytick.direction': 'out',
                  'savefig.bbox': 'tight',
                  'axes.unicode_minus': False,
                  'text.usetex': False}
        rcParams.update(params)

        # Set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon
        ds=xr.open_dataset(f'{self.casedir}/output/{key}_{xitem}.nc')
        # Extract variables
        lat = ds.lat.values
        lon = ds.lon.values
        lat, lon = np.meshgrid(lat[::-1], lon)

        var = ds[xitem].transpose("lon", "lat")[:, ::-1].values

        fig = plt.figure()
        M = Basemap(projection='cyl', llcrnrlat=self.min_lat, urcrnrlat=self.max_lat,
                    llcrnrlon=self.min_lon, urcrnrlon=self.max_lon, resolution='l')

        M.drawmapboundary(fill_color='white', zorder=-1)
        M.fillcontinents(color='0.8', lake_color='white', zorder=0)
        M.drawcoastlines(color='0.6', linewidth=0.1)

        loc_lon, loc_lat = M(lon, lat)

        cs = M.contourf(loc_lon, loc_lat, var, cmap=colormap, norm=normalize, levels=levels, extend='both')
        cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
        cb = fig.colorbar(cs, cax=cbaxes, ticks=levels, orientation='horizontal', spacing='uniform')
        cb.solids.set_edgecolor("face")
        cb.set_label('%s' % (xitem), position=(0.5, 1.5), labelpad=-35)
        plt.savefig(f'{self.casedir}/output/{key}_{xitem}.png', format='png', dpi=300)
        plt.close()


class Validation_stn:
    def __init__(self,info):
        self.name = 'Validation_plot'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        self.__dict__.update(info)
        if isinstance(self.sim_varname, str): self.sim_varname = [self.sim_varname]
        if isinstance(self.ref_varname, str): self.ref_varname = [self.ref_varname]

        print ('Validation processes starting!')
        print("=======================================")
        print(" ")
        print(" ")  

    def make_validation(self):
        #read station information
        stnlist  =f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist,header=0)

        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        for iik in range(len(station_list['ID'])):
            sim=xr.open_dataset(f"{self.casedir}/tmp/sim/sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[self.sim_varname]
            if self.sim_model=='CoLM':
                if self.sim_varname == 'H2OSOI':
                    sim=sim[:,0,:,:]
                elif self.sim_varname == 'QVEGT':
                    sim=sim*86400.
                else:
                    sim=sim
            ref=xr.open_dataset(f"{self.casedir}/tmp/ref/ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[self.ref_varname]
            p2=metrics_stn(sim.values,ref.values)
            for metric in self.metrics:
                if metric == 'pc_bias':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.pc_bias()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'apb':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.apb()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'RMSE':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.rmse()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'ubRMSE':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.ubRMSE()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'mae':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.mae()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'bias':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.bias()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'L':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.L()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'correlation':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.correlation()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'corrlation_R2':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.corrlation_R2()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'NSE':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.NSE()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'KGE':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.KGE()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'KGESS':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.KGESS()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'index_agreement':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.index_agreement()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'kappa_coeff':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.NSE()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'nBiasScore':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.nBiasScore()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                elif metric == 'nRMSEScore':
                    try:
                        station_list.loc[iik, f'{metric}']=p2.nRMSEScore()
                    except:
                        station_list.loc[iik, f'{metric}']=-9999.0
                else:
                    print('No such metric')
                    sys.exit(1)
                self.plot_stn(sim.squeeze(),obs.squeeze(),station_list['ID'][iik],self.ref_varname, float(station_list['RMSE'][iik]), float(station_list['KGE'][iik]),float(station_list['correlation'][iik]))
        print ('Comparison dataset prepared!')
        print("=======================================")
        print(" ")
        print(" ")  
        print(f"send {self.ref_varname} validation to {self.casedir}/{self.ref_varname}_metric.csv")
        station_list.to_csv(f'{self.casedir}/output/{self.ref_varname}_metric.csv',index=False)

    def plot_stn(self,sim,obs,ID,key,RMSE,KGESS,correlation):
        from pylab import rcParams
        import matplotlib
        import matplotlib.pyplot as plt
        ### Plot settings
        font = {'family' : 'DejaVu Sans'}
        #font = {'family' : 'Myriad Pro'}
        matplotlib.rc('font', **font)

        params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
        rcParams.update(params)
        
        legs =['obs','sim']
        lines=[1.5, 1.5]
        alphas=[1.,1.]
        linestyles=['solid','dotted']
        colors=['g',"purple"]
        fig, ax = plt.subplots(1,1,figsize=(10,5))
 
        obs.plot.line (x='time', label='obs', linewidth=lines[0], linestyle=linestyles[0], alpha=alphas[0],color=colors[0]                ) 
        sim.plot.line (x='time', label='sim', linewidth=lines[0], linestyle=linestyles[1], alpha=alphas[1],color=colors[1],add_legend=True) 
        #set ylabel to be the same as the variable name
        ax.set_ylabel(key, fontsize=18)        
        #ax.set_ylabel(f'{obs}', fontsize=18)
        ax.set_xlabel('Date', fontsize=18)
        ax.tick_params(axis='both', top='off', labelsize=16)
        ax.legend(loc='upper right', shadow=False, fontsize=14)
        #add RMSE,KGE,correlation in two digital to the legend in left top
        ax.text(0.01, 0.95, f'RMSE: {RMSE:.2f} \n R: {correlation:.2f} \n KGESS: {KGESS:.2f} ', transform=ax.transAxes, fontsize=14,verticalalignment='top')
        plt.tight_layout()
        plt.savefig(f'{self.casedir}/tmp/plt/{key[0]}_{ID}_timeseries.png')
        plt.close(fig)

    def plot_stn_map(self, stn_lon, stn_lat, metric, cmap, norm, ticks,key,varname):
        from pylab import rcParams
        from mpl_toolkits.basemap import Basemap
        import matplotlib
        import matplotlib.pyplot as plt
        ### Plot settings
        font = {'family' : 'DejaVu Sans'}
        #font = {'family' : 'Myriad Pro'}
        matplotlib.rc('font', **font)

        params = {'backend': 'ps',
          'axes.labelsize': 12,
          'grid.linewidth': 0.2,
          'font.size': 15,
          'legend.fontsize': 12,
          'legend.frameon': False,
          'xtick.labelsize': 12,
          'xtick.direction': 'out',
          'ytick.labelsize': 12,
          'ytick.direction': 'out',
          'savefig.bbox': 'tight',
          'axes.unicode_minus': False,
          'text.usetex': False}
        rcParams.update(params)
        fig = plt.figure()
        #set the region of the map based on self.Max_lat, self.Min_lat, self.Max_lon, self.Min_lon

        M = Basemap(projection='cyl',llcrnrlat=self.min_lat,urcrnrlat=self.max_lat,\
                    llcrnrlon=self.min_lon,urcrnrlon=self.max_lon,resolution='l')

        
        #fig.set_tight_layout(True)
        #M = Basemap(projection='robin', resolution='l', lat_0=15, lon_0=0)
        M.drawmapboundary(fill_color='white', zorder=-1)
        M.fillcontinents(color='0.8', lake_color='white', zorder=0)
        M.drawcoastlines(color='0.6', linewidth=0.1)
        #M.drawcountries(color='0.6', linewidth=0.1)
       # M.drawparallels(np.arange(-60.,60.,30.), dashes=[1,1], linewidth=0.25, color='0.5')
        #M.drawmeridians(np.arange(0., 360., 60.), dashes=[1,1], linewidth=0.25, color='0.5')
        loc_lon, loc_lat = M(stn_lon, stn_lat)
        cs = M.scatter(loc_lon, loc_lat, 15, metric, cmap=cmap, norm=norm, marker='.', edgecolors='none', alpha=0.9)
        cbaxes = fig.add_axes([0.26, 0.31, 0.5, 0.015])
        cb = fig.colorbar(cs, cax=cbaxes, ticks=ticks, orientation='horizontal', spacing='uniform')
        cb.solids.set_edgecolor("face")
        cb.set_label('%s'%(varname), position=(0.5, 1.5), labelpad=-35)
        plt.savefig(f'{self.casedir}/output/{key}_{varname}_validation.png',  format='png',dpi=400)
        plt.close()

    def make_plot_index(self):
        # read the data
        df = pd.read_csv(f'{self.casedir}/output/{self.ref_varname[0]}_metric.csv', header=0)
        # loop the keys in self.variables to get the metric output
        for metric in self.metrics:
            min_metric  = -999.0
            max_metric  = 100000.0
            print(df['%s'%(metric)])
            ind0 = df[df['%s'%(metric)]>min_metric].index
            data_select0 = df.loc[ind0]
            print(data_select0[data_select0['%s'%(metric)] < max_metric])
            ind1 = data_select0[data_select0['%s'%(metric)] < max_metric].index
            data_select = data_select0.loc[ind1]
            #if key=='discharge':
            #    #ind2 = data_select[abs(data_select['err']) < 0.001].index
            #    #data_select = data_select.loc[ind2]
            #    ind3 = data_select[abs(data_select['area1']) > 1000.].index
            #    data_select = data_select.loc[ind3]
            try:
                lon_select = data_select['ref_lon'].values
                lat_select = data_select['ref_lat'].values
            except:
                lon_select = data_select['sim_lon'].values
                lat_select = data_select['sim_lat'].values 
            plotvar=data_select['%s'%(metric)].values
            if metric == 'pc_bias':
                vmin=-100.0
                vmax= 100.0
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric)
            elif metric == 'KGE':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric)
            elif metric == 'KGESS':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd, self.ref_varname[0],metric)
            elif metric == 'NSE':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric)
            elif metric == 'correlation':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric) 
            elif metric == 'index_agreement':
                vmin=-1
                vmax= 1
                bnd = np.linspace(vmin, vmax, 11)
                cpool = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                cmap = colors.ListedColormap(cpool)
                norm = colors.BoundaryNorm(bnd, cmap.N)
                self.plot_stn_map(lon_select, lat_select, plotvar, cmap, norm, bnd,  self.ref_varname[0],metric)
    
    def make_validation_parallel(self,station_list,iik):
        sim=xr.open_dataset(f"{self.casedir}/tmp/sim/sim_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[self.sim_varname].to_array().squeeze()
        ref=xr.open_dataset(f"{self.casedir}/tmp/ref/ref_{station_list['ID'][iik]}_{station_list['use_syear'][iik]}_{station_list['use_eyear'][iik]}.nc")[self.ref_varname].to_array().squeeze()

        if self.sim_model=='CoLM':
            if self.sim_varname == 'H2OSOI':
                sim=sim[:,0,:,:]
            elif self.sim_varname == 'QVEGT':
                sim=sim*86400.
            elif self.sim_varname == 'f_fevpa':
                sim=sim*86400.

        #print(sim.values)
        #convert ref xarray.Dataset to numpy.ndarray
        p2=metrics_stn(sim.values,ref.values)

        row={}

        try:
            row['KGESS']=p2.KGESS()
        except:
            row['KGESS']=-9999.0
        try:
            row['RMSE']=p2.rmse()
        except:
            row['RMSE']=-9999.0 
        try:
            row['correlation']=p2.correlation()
        except:
            row['correlation']=-9999.0
        #print(metric)
        for metric in self.metrics:
            if metric == 'pc_bias':
                try:
                    row[f'{metric}']=p2.pc_bias()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'apb':
                try:
                    row[f'{metric}']=p2.apb()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'RMSE':
                try:
                    row[f'{metric}']=p2.rmse()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'ubRMSE':
                try:
                    row[f'{metric}']=p2.ubRMSE()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'mae':
                try:
                    row[f'{metric}']=p2.mae()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'bias':
                try:
                    row[f'{metric}']=p2.bias()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'L':
                try:
                    row[f'{metric}']=p2.L()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'correlation_R2':
                try:
                    row[f'{metric}']=p2.correlation_R2()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'correlation':
                try:
                    row[f'{metric}']=p2.correlation()
                except:
                    row[f'{metric}']=-9999.0                
            elif metric == 'NSE':
                try:
                    row[f'{metric}']=p2.NSE()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'KGE':
                try:
                    row[f'{metric}']=p2.KGE()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'KGESS':
                try:
                    row[f'{metric}']=p2.KGESS()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'index_agreement':
                try:
                    row[f'{metric}']=p2.index_agreement()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'kappa_coeff':
                try:
                    row[f'{metric}']=p2.NSE()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'nBiasScore':
                try:
                    row[f'{metric}']=p2.nBiasScore()
                except:
                    row[f'{metric}']=-9999.0
            elif metric == 'nRMSEScore':
                try:
                    row[f'{metric}']=p2.nRMSEScore()
                except:
                    row[f'{metric}']=-9999.0
            else:
                print(f'No such metric: {metric}')
                sys.exit(1)
        self.plot_stn(sim,ref,station_list['ID'][iik],self.ref_varname, float(row['RMSE']),float(row['KGESS']),float(row['correlation']))
        return row
        # return station_list
  
    def make_validation_P(self):
        stnlist  =f"{self.casedir}/stn_list.txt"
        station_list = pd.read_csv(stnlist,header=0)
        num_cores = os.cpu_count()  ##用来计算现在可以获得多少cpu核心。 也可以用multipocessing.cpu_count(),或者随意设定<=cpu核心数的数值
        shutil.rmtree(f'{self.casedir}/output',ignore_errors=True)
        #creat tmp directory
        os.makedirs(f'{self.casedir}/output', exist_ok=True)

        # loop the keys in self.variables
        # loop the keys in self.variables to get the metric output
        #for metric in self.metrics.keys():
        #    station_list[f'{metric}']=[-9999.0] * len(station_list['ID'])
        if self.ref_source.lower() == 'grdc':
            station_list['ref_lon'] = station_list['lon']
            station_list['ref_lat'] = station_list['lat']
        results=Parallel(n_jobs=num_cores)(delayed(self.make_validation_parallel)(station_list,iik) for iik in range(len(station_list['ID'])))
        station_list = pd.concat([station_list, pd.DataFrame(results)], axis=1)

        print ('simulation data prepared!')
        print("=======================================")
        print(" ")
        print(" ")  
        print(f"send {self.ref_varname[0]} validation to {self.casedir}/output/{self.ref_varname[0]}_metric.csv")
        station_list.to_csv(f'{self.casedir}/output/{self.ref_varname[0]}_metric.csv',index=False)
