import os, sys
import xarray as xr
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except:
    pass
'''
Data that I need:
    Yearly precip, temp max, temp min
'''

BASE_PATH = '/gss_gpfs_scratch/vandal.t/data-mining-project'
NEX_PATH = os.path.join(BASE_PATH, 'nex')
MONTHLY_PATH = os.path.join(BASE_PATH, 'ccsm4-monthly')
if not os.path.exists(MONTHLY_PATH):
    os.mkdir(MONTHLY_PATH)

def daily_to_monthly():
    scenarios = os.listdir(NEX_PATH)
    #scenarios = ['historical']
    for s in scenarios:
        if s == 'historical':
            min_year, max_year = 1950, 1999
        else:
            min_year, max_year = 2050, 2099
        s_path = os.path.join(NEX_PATH, s)
        variables = os.listdir(s_path)
        for v in variables:
            dss = []
            # load files in one by one and append a list, join them together then save
            print "Var: %s, Scenario: %s" % (v, s)
            v_path  = os.path.join(s_path, v)
            paths = sorted([os.path.join(v_path, p) for p in os.listdir(v_path)])
            for p in paths:
                y = int(os.path.basename(p).split("_")[5][:4])
                if (y >= min_year) and (y <= max_year):
                    ds = xr.open_dataset(p, engine='netcdf4')
                    dagg = ds.resample('MS', dim='time', how='mean')
                    dss.append(dagg)
                #if len(dss) >= 3:
                #    break
            dss = xr.concat(dss, dim='time')
            path = os.path.join(MONTHLY_PATH, '%s_%s_monthly.nc' % (s, v))
            dss.to_netcdf(path)

def build_anomaly_table():
    '''
    lat,lon,pr_1,pr_2,...,pr_n,tasmin_1,...,tasmin_n,tasmax_1,...,taxmax_n
    '''
    #scenarios = os.listdir(NEX_PATH)
    scenarios = ['historical']
    variables = sorted(os.listdir(os.path.join(NEX_PATH, scenarios[0])))
    for s in scenarios:
        files = []
        data = []
        for v in variables:
            files.append(os.path.join(MONTHLY_PATH, '%s_%s_monthly.nc' % (s, v)))

        ds = xr.open_mfdataset(files, engine='netcdf4')#.isel(time=range(12*5))
        climatology = ds.groupby('time.month').mean('time')
        anomalies = ds.groupby('time.month') - climatology
        idxs = np.where(ds['time.season'] == 'JJA')[0]
        anomalies = ds.isel(time=idxs)
        df = anomalies[variables].to_dataframe().unstack(2).dropna()
        print "Shape of table:",df.shape
        df.reset_index().to_csv(os.path.join(BASE_PATH, 'anomaly_JJA_%s.csv' % s), index=False)

def build_climatology_table():
    '''
    lat,lon,pr_1,pr_2,...,pr_n,tasmin_1,...,tasmin_n,tasmax_1,...,taxmax_n
    '''
    scenarios = os.listdir(NEX_PATH)
    variables = sorted(os.listdir(os.path.join(NEX_PATH, scenarios[0])))
    for s in scenarios:
        files = []
        data = []
        for v in variables:
            files.append(os.path.join(MONTHLY_PATH, '%s_%s_monthly.nc' % (s, v)))

        ds = xr.open_mfdataset(files, engine='netcdf4')#.isel(time=range(12*2))
        grouped = ds.groupby('time.month')
        mu = grouped.mean('time')
        std = grouped.std('time')
        df1 = mu[variables].to_dataframe().unstack(2).dropna()
        df2 = std[variables].to_dataframe().unstack(2).dropna()
        df = pd.concat([df1, df2], axis=1)
        print "Scenario: %s, Shape of table:" % s, df.shape
        df.reset_index().to_csv(os.path.join(BASE_PATH, 'CONUS_%s.csv' % s), index=False)

class ClimateData(object):
    def __init__(self, base_dir):
        self.scenarios = ['historical', 'rcp45', 'rcp85']
        #self.scenarios = ['historical',]
        self.base_dir = base_dir
        for s in self.scenarios:
            df = pd.read_csv(os.path.join(self.base_dir, 'CONUS_%s.csv' % s), skiprows=[1])
            scenario = _Scenario(df.values)
            setattr(self, s, scenario)

class _Scenario(object):
    def __init__(self, scenario):
        self._scenario = scenario
        self.latlon = self._scenario[:,:2]
        self.x = self._scenario[:,2:]

    def next_batch(self, batch_size):
        #randomly select rows
        idxs = np.random.choice(range(self.x.shape[0]), batch_size, replace=False)
        return self.x[idxs], self.latlon[idxs]

    def generate_epoch(self, batch_size):
        n_rows = self.x.shape[0]
        curr_row = 0
        idxs = range(n_rows)
        np.random.shuffle(idxs)
        while curr_row < n_rows:
            curr_idxs = idxs[curr_row:curr_row+batch_size]
            curr_row += batch_size
            yield self.x[curr_idxs], self.latlon[curr_idxs]

if __name__ == "__main__":
    #daily_to_monthly()
    #build_climatology_table()
    c = ClimateData(BASE_PATH)
    print c.historical.x.shape
