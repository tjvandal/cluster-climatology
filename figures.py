from preprocess import ClimateData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

def make_map(dataxray, ax, title='', vmin=None, vmax=None):
    data = dataxray.values
    latcorners = dataxray.index.values - 3./8
    loncorners = dataxray.columns.values #+ 0.25
    lat_0 = latcorners.mean()
    lon_0 = loncorners.mean()

    # create figure and axes instances

    # create polar stereographic Basemap instance.
    m = Basemap(projection='merc', lat_0=lat_0, lon_0=lon_0, \
                llcrnrlat=latcorners[0],urcrnrlat=latcorners[-1],\
                llcrnrlon=loncorners[0],urcrnrlon=loncorners[-1],\
                resolution='l', ax=ax)
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    # draw parallels.
    parallels = np.arange(0.,90,5.)
    #m.drawparallels(parallels,labels=[1,0,0,0],fontsize=8)
    # draw meridians
    meridians = np.arange(-180,0.,5)
    #m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8, rotation=30)

    ny = data.shape[0]; nx = data.shape[1]
    lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
    x, y = m(lons, lats) # compute map proj coordinates.
    cs = m.imshow(data, interpolation='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.axis('off')
    #cs = m.imshow(data, interpolation='nearest', cmap='bwr', vmin=vmin, vmax=vmax)

    # add colorbar.
    #cbar = m.colorbar(cs,location='right',pad="8%")
    #cbar.set_label('mm/day', fontsize=12, horizontalalignment='right')
    # add title
    ax.set_title(title, fontsize=12)

def example_figure():
    col = 8+12 # should be >= 2
    data = ClimateData('/gss_gpfs_scratch/vandal.t/data-mining-project')
    runs = ['historical', 'rcp45', 'rcp85']
    fig, axs = plt.subplots(3,1, figsize=(6, 12), facecolor='w', edgecolor='k', 
                               sharex=True, sharey=True)
    axs = axs.ravel()
    vmin = data.historical._scenario[:,col].min()
    vmax = data.historical._scenario[:,col].max()
    for i, r in enumerate(runs):
        d = getattr(data, r)
        arr = d._scenario[:,[0,1,col]]
        df = pd.DataFrame(arr, columns=['lat', 'lon', 'var'])
        df = pd.pivot_table(df, index='lat', columns='lon', values='var')
        make_map(df, axs[i], title=r.upper(), vmin=vmin, vmax=vmax)
    plt.savefig('figures/example_tmax.pdf')

def comparison_figure():
    scenarios = ['historical', 'rcp45', 'rcp85']
    models = ['kmeans', 'dec10']
    fig, axs = plt.subplots(3,2,figsize=(12,12))
    axs = axs.ravel()
    for i, s in enumerate(scenarios):
        for j, m in enumerate(models):
            k = i*len(models) + j
            print s, m, k
            if s == 'historical':
                years = '1950-1999'
            else:
                years = '2050-2099'
            f = 'output/%s_%s.csv' % (m, s)
            df = pd.read_csv(f, index_col=0)
            axs[k].imshow(df.values[::-1])
            axs[k].axis('off')
            axs[k].set_title(m.upper() + ' - ' + s.upper() + ' - ' + years)
    plt.savefig('figures/compare.pdf')
    plt.close()

def comparison_table():
    scenarios = ['historical', 'rcp45', 'rcp85']
    models = ['kmeans', 'dec10']
    dfs = dict()
    for j, m in enumerate(models):
        for i, s in enumerate(scenarios):
            f = 'output/%s_%s.csv' % (m, s)
            dfs['%s_%s' % (m,s)] = pd.read_csv(f, index_col=0)
            sim = (dfs['%s_historical' % m] != dfs['%s_%s' % (m,s)])
            mask = pd.notnull(dfs['%s_historical'% m])
            mask = mask.replace(False, np.nan) * 1.
            diff = (dfs['%s_historical' % m] != dfs['%s_%s' % (m,s)])*mask
            change = np.nanmean(diff.values)
            print m, s, change


if __name__ == "__main__":
    #example_figure()
    #comparison_figure()
    comparison_table()
