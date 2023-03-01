#!/usr/bin/env python
# coding: utf-8

# %%
from matplotlib import pyplot as plt
import cmocean
import palettable

# %%
import pathlib
import argparse

# %%
from datetime import date, timedelta

# %%
import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
import cartopy.feature as cfeature

# %%
import OISST

# %% uncomment and edit this for testing interactively
# domain = "NZ"
# ipath = "/media/nicolasf/END19101/data/OISST/daily"
# clim_path = "/home/nicolasf/operational/OISST_indices/outputs/"
# fig_path = "/home/nicolasf/operational/OISST_indices/figures/"
# ndays_agg = 30
# ndays_back = 365 * 10
# quantile = .9
# lag = 5

# ipath = pathlib.Path(ipath).joinpath(domain)
# clim_path = pathlib.Path(clim_path).joinpath(domain)
# fig_path = pathlib.Path(fig_path)

# %%
parser = argparse.ArgumentParser(prog = 'OISST_realtime_NZ_maps.py',
                                description = 'Maps the SST anomalies around NZ along with the MHWs conditions', 
                                epilog = 'Relies on `update_OISST.py` to be run first\n')

# %%
parser.add_argument('-l', '--lag', type=int, default=0,
                    help="lag (in days) WRT to realtime, default 0")

parser.add_argument('-q', '--quantile', type=float, default=0.9,
                    help="The quantile threshold used to define heatwave conditions, default 0.9 (90th percentile)")

parser.add_argument('-d', '--domain', type=str, default='NZ',
                    help="The domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ'")

parser.add_argument('-i', '--ipath', type=str, default='/media/nicolasf/END19101/data/OISST/daily',
                    help="The root path where the downloaded OISST datasets are, default to '/media/nicolasf/END19101/data/OISST/daily'")

parser.add_argument('-c', '--clim_path', type=str, default='/home/nicolasf/operational/OISST_indices/outputs/',
                    help="The path to the zarr files containing the percentile climatologies, default to '/home/nicolasf/operational/OISST_indices/outputs/'")

parser.add_argument('-f', '--fig_path', type=str, default='/home/nicolasf/operational/OISST_indices/figures/',
                    help="The path to where the figures are saved, default to '/home/nicolasf/operational/OISST_indices/figures/'")
                    
parser.add_argument('-n', '--ndays_agg', type=int, default=1,
                    help="The averaging period in days, can be in [1, 7, 30] currently")

parser.add_argument('-b', '--ndays_back', type=int, default=3650,
                    help="The number of days to look back for the calculation of the cumulative MHWs conditions, default 10 years (3650 days)")

# %% 
args = parser.parse_args()


# %% get the arguments and cast to the correct type

lag = args.lag
quantile = float(args.quantile)
domain = args.domain
ipath = pathlib.Path(args.ipath).joinpath(domain)
clim_path = pathlib.Path(args.clim_path).joinpath(domain)
fig_path = pathlib.Path(args.fig_path)
ndays_agg = int(args.ndays_agg)
ndays_back = int(args.ndays_back)


# %%
current_date = date.today()

# %%
last_date = current_date - timedelta(days=lag)

# %%
first_date = last_date - timedelta(days=ndays_back)

# %%
years_to_get = np.unique(np.arange(first_date.year, last_date.year + 1))


# %%
lfiles = [ipath.joinpath(f"sst.day.mean.{year}.nc") for year in years_to_get]

# %%
lfiles.sort()

# %%
dset = xr.open_mfdataset(lfiles, parallel=True, combine="by_coords", engine='netcdf4') 

# %%
dset = dset.sortby('time')

#%% 
dset = dset.sel(time=slice(None, last_date))

#%% 
dset

# %%
if ndays_agg > 1:

    dset = dset.rolling({"time": ndays_agg}, min_periods=ndays_agg, center=False).mean(
        "time"
    )

    dset = dset.isel(time=slice(ndays_agg + 1, None))

# %%
first_day = pd.to_datetime(dset.time.data[0])
last_day = pd.to_datetime(dset.time.data[-1])

# %%
dset = dset.convert_calendar("noleap")

# %%
clim = xr.open_zarr(
    clim_path.joinpath(f"{domain}_OISST_{ndays_agg}days_climatology_15_window.zarr")
)


# %%
anoms = dset.groupby(dset.time.dt.dayofyear) - clim["average"]

# %%
clim_repeat = clim.sel(dayofyear=dset.time.dt.dayofyear)


# %%
mask = dset["sst"].where(dset["sst"] >= clim_repeat["quantiles"].sel(quantile=quantile))
mask = mask.where(np.isnan(mask), other=1)

#%% 
mask

# %%
dataarray_anoms = anoms["sst"].isel(time=-1)

# %%
dataarray_raw = dset["sst"].isel(time=-1)

# %%
dataarray_mask = mask.isel(time=-1)

# %%
dataarray_anoms = OISST.interpolate_NaN_da(dataarray_anoms)

# %%
dataarray_raw = OISST.interpolate_NaN_da(dataarray_raw)

# %%
cmap = cmocean.cm.balance

# %%
f, ax = OISST.plot_SST_map(
    dataarray_anoms,
    dataarray_mask,
    ndays_agg=ndays_agg,
    levels=np.arange(-3, 3 + 0.25, 0.25),
)


# %%
f.savefig(
    fig_path.joinpath(f"proto_OISST_anoms_{ndays_agg:02d}_days_to_{last_day:%Y-%m-%d}.png"),
    dpi=200,
    bbox_inches="tight",
    facecolor="w",
)

# %%
heatwaves_days = mask.copy()

# %%
heatwaves_days = heatwaves_days.fillna(0)

# %%
heatwaves_days = 1 - heatwaves_days

# %%
heatwaves_days_consecutive = heatwaves_days.cumsum(dim="time", keep_attrs=True)


# %%
heatwaves_days_consecutive["time"] = (
    ("time"),
    np.arange(len(heatwaves_days.time))[::-1],
)


# %%
heatwaves_days_consecutive = heatwaves_days_consecutive.idxmax(dim="time")


# %%
cmap = palettable.scientific.sequential.Bilbao_20.mpl_colormap


# %%
states_provinces = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="10m",
    facecolor="none",
)

lakes = cfeature.NaturalEarthFeature("physical", "lakes", "10m")
land = cfeature.NaturalEarthFeature("physical", "land", "10m")

# %%
cbar_kwargs = {"shrink": 0.8, "label": "nb. of periods"}

# %%
f, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)),
)

heatwaves_days_consecutive.plot(
    ax=ax,
    levels=np.arange(0, 200, 10),
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    cbar_kwargs=cbar_kwargs,
)

ax.add_feature(land, facecolor="gainsboro")
ax.add_feature(lakes, facecolor="b", edgecolor="b", alpha=0.2)
ax.add_feature(states_provinces, edgecolor="k", linewidth=1)
ax.coastlines("10m", linewidth=1)

ax.set_title(
    f"NIWA Marine Heat Waves tracking: {ndays_agg} day(s) to {last_day:%Y-%m-%d}\nNumber of consecutive {ndays_agg} day(s) periods\nabove {int(quantile*100):02d}th percentile",
    fontsize=14,
)

# %%
f.savefig(
    fig_path.joinpath(f"proto_OISST_heatwave_days_{ndays_agg:02d}_days_to_{last_day:%Y-%m-%d}.png"),
    dpi=200,
    bbox_inches="tight",
    facecolor="w",
)
# %%
