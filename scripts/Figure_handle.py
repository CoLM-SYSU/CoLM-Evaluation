import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import colors
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

class FigureHandler():
    def __init__(self, info):
        self.name = 'plotting'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.__dict__.update(info)

        os.makedirs(self.casedir+'/output/', exist_ok=True)

        print('Plotting processes starting!')
        print("=======================================")
        print(" ")
        print(" ")

    def make_geo_plot_index(self):
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
            self.plot_geo_map(cmap, norm, key, bnd,metric)
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
            self.plot_geo_map(cmap, norm, key, bnd,score)
        print("=======================================")

    def plot_geo_map(self, colormap, normalize, key, levels, xitem, **kwargs):
        # Plot settings
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

    def plot_stn(self,sim,obs,ID,key,RMSE,KGESS,correlation):        
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

    def make_stn_plot_index(self):
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
    
    