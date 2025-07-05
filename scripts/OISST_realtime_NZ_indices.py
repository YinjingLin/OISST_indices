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
print(f"executing OISST_realtime_NZ_indices.py with {str(sys.executable)}\n")

parser = argparse.ArgumentParser(prog = 'OISST_realtime_NZ_indices.py',
                                description = 'Plot the time-series of OISST V2 SST anomalies for the 6 NZ coastal regions', 
                                epilog = 'Relies on `update_OISST.py` to be run first\n')

parser.add_argument('-d', '--domain', type=str, default='NZ',
                    help="The domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ' and should not change ...")

parser.add_argument('-i', '--ipath', type=str, default='/media/nicolasf/END19101/data/OISST/daily',
                    help="The root path where the downloaded OISST datasets are, default to '/media/nicolasf/END19101/data/OISST/daily'")

parser.add_argument('-c', '--clim_path', type=str, default='/home/nicolasf/operational/OISST_indices/outputs/',
                    help="The path to the zarr files containing the climatologies, default to '/home/nicolasf/operational/OISST_indices/outputs/'")

parser.add_argument('-s', '--shapes_path', type=str, default='/home/nicolasf/operational/OISST_indices/data/shapefiles/',
                    help="The path to the shapefiles used to delineates the 6 NZ coastal regions, default to '/home/nicolasf/operational/OISST_indices/data/shapefiles/'")

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
shapes_path = pathlib.Path(args.shapes_path)
fig_path = pathlib.Path(args.fig_path)
csv_path = pathlib.Path(args.csv_path)
ndays_agg = int(args.ndays_agg)
nmonths_back = int(args.nmonths_back)

# %% 
# this is hardcoded 
NZ_regions = ["NNI", "WNI", "ENI", "NSI", "WSI", "ESI"]


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

# %% Open the files safely
dset = xr.open_mfdataset(lfiles, parallel=False, combine="by_coords")

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

# %% open the climatologies 
clim = xr.open_zarr(
    clim_path.joinpath(f"{domain}_OISST_{ndays_agg}days_climatology_15_window.zarr")
)

# %% calculate the climatology 
anoms = dset.groupby(dset.time.dt.dayofyear) - clim["average"]

# %% repeat the climo  
clim_repeat = clim.sel(dayofyear=dset.time.dt.dayofyear)

# %% get the masks 
l_masks = []

for NZ_region in NZ_regions:

    mask = OISST.make_mask_from_shape(
        shapes_path.joinpath(f"{NZ_region}_buffered_50km.shp"),
        dset,
        to_crs="EPSG:4326",
        mask_name=NZ_region,
    )

    l_masks.append(mask)

NZ_regions_masks = xr.merge(l_masks)

# %% merge the dataset containing the anomalies with the masks 
anoms = anoms.merge(NZ_regions_masks)

# %% same for the dataset containing the raw values 
dset = dset.merge(NZ_regions_masks)

# %% same for the climatology 
clim_repeat = clim_repeat.merge(NZ_regions_masks)

# %% now calculate the average anomalies 

anoms_ts = []

for NZ_region in NZ_regions:

    anom_ts = (anoms["sst"] * anoms[NZ_region]).mean(["lat", "lon"])

    anom_ts.name = f"SST_anomalies_{NZ_region}"

    anoms_ts.append(anom_ts)

anoms_ts = xr.merge(anoms_ts)

# %% and convert back to standard calendar 
# anoms_ts = anoms_ts.interp_calendar(standard_calendar) # removes this, source of the problem with last day being a NaN

# %% convert to dataframe 
anoms_ts = anoms_ts.to_pandas()

# %% drop the day of year column
anoms_ts = anoms_ts.drop("dayofyear", axis=1)

# %%% fix the calendar issue 
anoms_ts_index = [datetime(d.year, d.month, d.day) for d in anoms_ts.index]
anoms_ts_index = pd.DatetimeIndex(anoms_ts_index)
anoms_ts.index = anoms_ts_index
anoms_ts = anoms_ts.reindex(standard_calendar)
anoms_ts = anoms_ts.interpolate()

# %% fix the column names 
anoms_ts.columns = NZ_regions

# %% export to CSV 

csv_path = csv_path.joinpath(domain)

csv_path.mkdir(parents=True, exist_ok=True) 

anoms_ts.to_csv(csv_path.joinpath(f"{domain}_time-series_{ndays_agg}_days_anomalies_to_{last_day:%Y-%m-%d}.csv"))

# %% print the last 10 days
print(anoms_ts.tail(10))

# %% plot the time-series 
f, axes = plt.subplots(
    nrows=len(NZ_regions), figsize=(10, 18), sharex=True, sharey=True
)

plt.subplots_adjust(hspace=0.4)

for i, NZ_region in enumerate(NZ_regions):

    ax = axes[i]

    df = anoms_ts.loc[:, NZ_region]

    ax.fill_between(
        df.index, 0, df.values, df.values > 0, interpolate=True, color="coral"
    )
    ax.fill_between(
        df.index, 0, df.values, df.values <= 0, interpolate=True, color="steelblue"
    )

    ax.plot(df.index, df.values, color="k", lw=0.5)

    ax.grid(ls=":")

    title = (
        r"$\bf{"
        + NZ_region
        + "}$"
        + f" OISST V2 {ndays_agg} day(s) anomalies to {last_day:%Y-%m-%d}\nMin: {df.min():+4.2f}°C ({df.idxmin():%Y-%m-%d}), Max: {df.max():+4.2f}°C ({df.idxmax():%Y-%m-%d}), latest: {df.iloc[-1]:+4.2f}°C"
    )

    ax.set_title(title, loc="left")

    ax.axhline(0, color="0.8", zorder=-1)

    ax.axvline(df.idxmin(), color="b", alpha=0.5)
    ax.axvline(df.idxmax(), color="r", alpha=0.5)

    ax.set_xlim(first_day, last_day)

# %% save the figure
f.savefig(
    fig_path.joinpath(
        f"prototype_NZ_coastal_indices_{ndays_agg}days_agg_to_{last_day:%Y%m%d}.png"
    ),
    dpi=200,
    bbox_inches="tight",
    facecolor="w",
)

# %% close the datasets
dset.close()
clim.close()
anoms.close()
