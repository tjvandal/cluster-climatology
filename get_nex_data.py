import boto
import os

save_dir = "/gss_gpfs_scratch/vandal.t/data-mining-project/nex"

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def download():
    # 's3://nasanex/NEX-DCP30/BCSD/historical/mon/atmos/pr/r1i1p1/v1.0/CONUS/pr_amon_BCSD_historical_r1i1p1_CONUS_CCSM4_195001-195412.nc'
    # s3://nasanex/LOCA/CCSM4/16th/historical/r6i1p1/pr/pr_day_CCSM4_historical_r6i1p1_19500101-19501231.LOCA_2016-04-02.16th.nc
    variables = ['pr', 'tasmax', 'tasmin']
    runs = ['historical', 'rcp45', 'rcp85']
    s3_dir = 'LOCA/CCSM4/16th/%s/r6i1p1/%s/'

    # initialize boto
    conn = boto.connect_s3()
    bucket = conn.get_bucket("nasanex")
    count = 0
    for r in runs:
        run_path = os.path.join(save_dir, r)
        check_dir(run_path)
        for v in variables:
            v_path = os.path.join(run_path, v)
            check_dir(v_path)
            for key in bucket.list(s3_dir % (r, v)):
                count += 1
                print "Downloading File Number:", count
                fname = os.path.join(v_path,os.path.basename(key.name))
                key.get_contents_to_filename(fname)

if __name__ == "__main__":
    download()
