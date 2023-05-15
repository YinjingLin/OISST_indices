#!/usr/bin/env python
# coding: utf-8

# %% 
from matplotlib import pyplot as plt
from matplotlib import rc

# %%
import sys
import pathlib
import argparse

# %% 
from datetime import datetime, timedelta
from dateparser import parse

# %% 
import numpy as np
import pandas as pd
import xarray as xr

# %% 
import OISST 

# %% 
print(f"executing OISST_realtime_IOD_indices.py with {str(sys.executable)}\n")

parser = argparse.ArgumentParser(prog = 'OISST_realtime_IOD_indices.py',
                                description = 'Plot the time-series of OISST V2 SST anomalies for the IOD index', 
                                epilog = 'Relies on `update_OISST.py` to be run first\n')

parser.add_argument('-d', '--domain', type=str, default='Ninos',
                    help="The domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'IOD' and should not change ...")

parser.add_argument('-i', '--ipath', type=str, default='/media/nicolasf/END19101/data/OISST/daily',
                    help="The root path where the downloaded OISST datasets are, default to '/media/nicolasf/END19101/data/OISST/daily'")

parser.add_argument('-c', '--clim_path', type=str, default='/home/nicolasf/operational/OISST_indices/outputs/',
                    help="The path to the zarr files containing the climatologies, default to '/home/nicolasf/operational/OISST_indices/outputs/'")

parser.add_argument('-f', '--fig_path', type=str, default='/home/nicolasf/operational/OISST_indices/figures/',
                    help="The path to where the figures are saved, default to '/home/nicolasf/operational/OISST_indices/figures/'")

parser.add_argument('-o', '--csv_path', type=str, default='/home/nicolasf/operational/OISST_indices/outputs/',
                    help="The path to where the csv files containing the time-series are saved '/home/nicolasf/operational/OISST_indices/outputs/'")
  
parser.add_argument('-n', '--ndays_agg', type=int, default=1,
                    help="The averaging period in days, can be in [1, 7, 30] currently")

parser.add_argument('-m', '--nmonths_back', type=int, default=36,
                    help="The number of months to look back, default 36")


args = parser.parse_args()


# %% get the arguments and cast to the correct type

domain = args.domain
ipath = pathlib.Path(args.ipath).joinpath(domain)
clim_path = pathlib.Path(args.clim_path).joinpath(domain)
fig_path = pathlib.Path(args.fig_path)
csv_path = pathlib.Path(args.csv_path)
ndays_agg = int(args.ndays_agg)
nmonths_back = int(args.nmonths_back)

# %% 
# get the current date 
current_date = datetime.utcnow()

# %% 
# get the first day of the period 
first_day = parse(f"{nmonths_back} months ago")

# %% 
first_day = first_day - timedelta(days=first_day.day - 1)

# %% get the years 
years_to_get = np.unique(np.arange(first_day.year, current_date.year + 1))

# %% list the files 
lfiles = [ipath.joinpath(f"sst.day.mean.{year}.nc") for year in years_to_get]

# %% 
lfiles.sort()

# %% open the files 
try:
    dset = xr.open_mfdataset(lfiles, parallel=True, combine="by_coords")
except KeyError:
    dset = xr.open_mfdataset(lfiles, combine="by_coords")

# %% calculate the rolling averages if needed 
if ndays_agg > 1:

    dset = dset.rolling({"time": ndays_agg}, min_periods=ndays_agg, center=False).mean(
        "time"
    )

    dset = dset.isel(time=slice(ndays_agg + 1, None))

# %% 
first_day = pd.to_datetime(dset.time.data[0])
last_day = pd.to_datetime(dset.time.data[-1])

# %% ge the calendar 
standard_calendar = pd.date_range(start=first_day, end=last_day, freq="D")

#%% convert to no leap years calendar 
dset = dset.convert_calendar("noleap")

# %% calculate the Ninos indices
dset = OISST.calculates_IOD_nodes(dset, IOD_node='all')

# %% open the climatologies 
clim = xr.open_zarr(
    clim_path.joinpath(f"{domain}_OISST_{ndays_agg}days_climatology_15_window.zarr")
)

# %% remove the average 
anoms = dset.groupby(dset.time.dt.dayofyear) - clim["average"]

# %% divide by the standard deviation 
anoms_std = anoms.groupby(dset.time.dt.dayofyear) / clim["std"]

# %% repeat the climo , not used so drop ? 
# clim_repeat = clim.sel(dayofyear=dset.time.dt.dayofyear)

# %% Now interpolate over the standard calendar
anoms_std = anoms_std.interp_calendar(standard_calendar)

# %% 
anoms_ts = anoms_std['sst'].to_pandas().T

# %% IOD 
anoms_ts = anoms_ts.assign(IOD = anoms_ts.loc[:,'IOD_West'] - anoms_ts.loc[:,'IOD_East'])

# %% export to CSV 

csv_path = csv_path.joinpath(domain)

csv_path.mkdir(parents=True, exist_ok=True) 

anoms_ts.to_csv(csv_path.joinpath(f"{domain}_time-series_{ndays_agg}_days_anomalies_to_{last_day:%Y-%m-%d}.csv"))

# %% print the last 10 days
print(anoms_ts.tail(10))

# %% plot the time-series
f, axes = plt.subplots(
    nrows=len(anoms_ts.columns), figsize=(10, 15), sharex=True, sharey=True
)

plt.subplots_adjust(hspace=0.35)

for i, region in enumerate(anoms_ts.columns):

    ax = axes[i]

    df = anoms_ts.loc[:, region]
        
    ax.fill_between(
        df.index, 0, df.values, df.values > 0, interpolate=True, color="coral"
    )
    ax.fill_between(
        df.index, 0, df.values, df.values <= 0, interpolate=True, color="steelblue"
    )

    ax.plot(df.index, df.values, color="k", lw=0.5)
    
    ax.grid(ls=":")

    title = r"$\bf{" + region.replace('_','') + "}$" + f", OISST V2 {ndays_agg} days standardized anomalies to {last_day:%Y-%m-%d}\nMin: {df.min():+4.2f} std ({df.idxmin():%Y-%m-%d}), Max: {df.max():+4.2f} std ({df.idxmax():%Y-%m-%d}), latest: {df.iloc[-1]:+4.2f} std"

    ax.set_title(title, loc="left")

    ax.axhline(0, color="0.8", zorder=-1)

    ax.axvline(df.idxmin(), color="b", alpha=0.5)
    ax.axvline(df.idxmax(), color="r", alpha=0.5)

    ax.set_xlim(first_day, last_day)

# %% 
print(f"saving figure in {str(fig_path)}")


# %% save the figure
f.savefig(
    fig_path.joinpath(
        f"prototype_IOD_{ndays_agg}days_agg_to_{last_day:%Y%m%d}.png"
    ),
    dpi=200,
    bbox_inches="tight",
    facecolor="w",
)

# %% close the datasets
dset.close()
clim.close()
anoms.close()