import os
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
TMP_PATH = os.path.join(BASE_PATH, 'tmp-all')
if not os.path.exists(TMP_PATH):
    os.mkdir(TMP_PATH)

def main():
    scenarios = os.listdir(NEX_PATH)
    for s in scenarios:
        s_path = os.path.join(NEX_PATH, s)
        variables = os.listdir(s_path)
        for v in variables:
            dss = []
            # load files in one by one and append a list, join them together then save
            print "Var: %s, Scenario: %s" % (v, s)
            v_path  = os.path.join(s_path, v)
            paths = sorted([os.path.join(v_path, p) for p in os.listdir(v_path)])
            for p in paths:
                ds = xr.open_dataset(p, engine='netcdf4')
                dagg = ds.resample('A', dim='time', how='mean')
                dss.append(dagg)
            dss = xr.concat(dss, dim='time')
            tmp_path = os.path.join(TMP_PATH, '%s_%s.nc' % (s, v))
            dss.to_netcdf(tmp_path)

def build_table():
    '''
    lat,lon,pr_1,pr_2,...,pr_n,tasmin_1,...,tasmin_n,tasmax_1,...,taxmax_n
    '''
    scenarios = os.listdir(NEX_PATH)
    variables = sorted(os.listdir(os.path.join(NEX_PATH, scenarios[0])))
    for s in scenarios:
        files = []
        data = []
        for v in variables:
            files.append(os.path.join(TMP_PATH, '%s_%s.nc' % (s, v)))
        ds = xr.open_mfdataset(files, engine='netcdf4')
        df = ds[variables].to_dataframe().unstack(2).dropna()
        df.reset_index().to_csv(os.path.join(BASE_PATH, '%s.csv' % s), index=False)
        print s, v

class ClimateData(object):
    def __init__(self, base_dir):
        self.scenarios = ['historical', 'rcp45', 'rcp85']
        self.base_dir = base_dir
        for s in self.scenarios:
            df = pd.read_csv(os.path.join(self.base_dir, '%s.csv' % s), skiprows=[1])
            scenario = _Scenario(df.values)
            setattr(self, s, scenario)

class _Scenario(object):
    def __init__(self, scenario):
        self._scenario = scenario
        self.latlon = self._scenario[:,:2]
        self.x = self._scenario[:,2:]

    def next_batch(self, batch_size):
        #randomly select rows
        idxs = np.random.choice(range(self.x.shape[0]), batch_size)
        return self.x[idxs], self.latlon[idxs]

if __name__ == "__main__":
    # main()
    build_table()
    #c = ClimateData(BASE_PATH)
    #print c.historical.next_batch(10)
