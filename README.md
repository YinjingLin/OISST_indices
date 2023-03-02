## Sea Surface Temperature (SST) indices and maps derived from the daily OISST v2 product 

This repository contains Jupyter notebooks, Python scripts and a module (in [OISST.py](OISST.py)) to calculate, maps and plots a series 
of diagnostics related to Sea Surface Temperature (SST) anomalies and Marine Heat Wave (MHW) conditions around Aotearoa New Zealand.

Author: [Nicolas Fauchereau](mailto:Nicolas.Fauchereau@niwa.co.nz)

The first port of call is the [scripts](https://github.com/nicolasfauchereau/OISST_indices/tree/main/scripts) directory. 

It contains 3 scripts: 

- [update_OISST.py](https://github.com/nicolasfauchereau/OISST_indices/blob/main/scripts/update_OISST.py)   
  
update the OISST dataset locally for a given geographical domain 

- [OISST_realtime_NZ_maps.py](https://github.com/nicolasfauchereau/OISST_indices/blob/main/scripts/OISST_realtime_NZ_maps.py) 

Plots the maps of SST anomalies with MHW conditions stippled, and the maps showing the number of consecutive periods (1, 7, 30 days) to 
date with MHW conditions

- [OISST_realtime_NZ_indices.py](https://github.com/nicolasfauchereau/OISST_indices/blob/main/scripts/OISST_realtime_NZ_indices.py) 

Plots the time-series of the Aotearoa New Zealand 6 coastal SST indices (see map below)

Below is the help for these scripts 

- *update_OISST.py* 

```
usage: update_OISST.py [-h] [-y YEAR] [-o OPATH] [-d DOMAIN] [-t TRYDAP]

Download the yearly OISST v2 netcdfs from the PSL

optional arguments:
  -h, --help            show this help message and exit
  -y YEAR, --year YEAR  The year to download the OISST v2 netcdf file for Note: - root URL is
                        https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/ - The filename pattern is sst.day.mean.${year}.v2.nc
  -o OPATH, --opath OPATH
                        The root path where to download the OISST datasets, default to '/media/nicolasf/END19101/data/OISST/daily/'
  -d DOMAIN, --domain DOMAIN
                        the domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ'
  -t TRYDAP, --tryDAP TRYDAP
                        Whether to try the PSL DAP server, probably better to leave it at 0 (default, which means False)

Always check data availability at https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/

```

- *OISST_realtime_NZ_maps.py* 

```
usage: OISST_realtime_NZ_maps.py [-h] [-l LAG] [-q QUANTILE] [-d DOMAIN] [-i IPATH] [-c CLIM_PATH] [-f FIG_PATH] [-n NDAYS_AGG] [-b NDAYS_BACK]

Maps the SST anomalies around NZ along with the MHWs conditions

optional arguments:
  -h, --help            show this help message and exit
  -l LAG, --lag LAG     lag (in days) WRT to realtime, default 0
  -q QUANTILE, --quantile QUANTILE
                        The quantile threshold used to define heatwave conditions, default 0.9 (90th percentile)
  -d DOMAIN, --domain DOMAIN
                        The domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ'
  -i IPATH, --ipath IPATH
                        The root path where the downloaded OISST datasets are, default to '/media/nicolasf/END19101/data/OISST/daily'
  -c CLIM_PATH, --clim_path CLIM_PATH
                        The path to the zarr files containing the percentile climatologies, default to
                        '/home/nicolasf/operational/OISST_indices/outputs/'
  -f FIG_PATH, --fig_path FIG_PATH
                        The path to where the figures are saved, default to '/home/nicolasf/operational/OISST_indices/figures/'
  -n NDAYS_AGG, --ndays_agg NDAYS_AGG
                        The averaging period in days, can be in [1, 7, 30] currently
  -b NDAYS_BACK, --ndays_back NDAYS_BACK
                        The number of days to look back for the calculation of the cumulative MHWs conditions, default 10 years (3650 days)

Relies on `update_OISST.py` to be run first
```

- *OISST_realtime_NZ_indices.py* 

```
usage: OISST_realtime_NZ_indices.py [-h] [-d DOMAIN] [-i IPATH] [-c CLIM_PATH] [-s SHAPES_PATH] [-f FIG_PATH] [-n NDAYS_AGG] [-m NMONTHS_BACK]

Plot the time-series of OISST V2 SST anomalies for the 6 NZ coastal regions

optional arguments:
  -h, --help            show this help message and exit
  -d DOMAIN, --domain DOMAIN
                        The domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ' and should not change ...
  -i IPATH, --ipath IPATH
                        The root path where the downloaded OISST datasets are, default to '/media/nicolasf/END19101/data/OISST/daily'
  -c CLIM_PATH, --clim_path CLIM_PATH
                        The path to the zarr files containing the climatologies, default to '/home/nicolasf/operational/OISST_indices/outputs/'
  -s SHAPES_PATH, --shapes_path SHAPES_PATH
                        The path to the shapefiles used to delineates the 6 NZ coastal regions, default to
                        '/home/nicolasf/operational/OISST_indices/data/shapefiles/'
  -f FIG_PATH, --fig_path FIG_PATH
                        The path to where the figures are saved, default to '/home/nicolasf/operational/OISST_indices/figures/'
  -n NDAYS_AGG, --ndays_agg NDAYS_AGG
                        The averaging period in days, can be in [1, 7, 30] currently
  -m NMONTHS_BACK, --nmonths_back NMONTHS_BACK
                        The number of months to look back, default 36

Relies on `update_OISST.py` to be run first
```  

<hr>

In the [notebooks](https://github.com/nicolasfauchereau/OISST_indices/tree/main/notebooks) directory are some notebooks mainly used to calculate the climatologies, which should only be needed to be run if calculating climatologies for another domain or another period. 

**calculate_OISST_climatologies.ipynb** calculates the climatological values for a particular domain, i.e. calculates average, std, and quantiles [0.1, 0.25, 0.5, 0.75, 0.9] for a given climatology period (currently 1991 - 2020). Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, currently can be ['NZ', 'Ninos', 'IOD']
    - `ndays` (int): The aggregation period (i.e. 1, 7 or 30 days)
    - `roll` (int): The buffer (in days) around the day of the year for the calculation of the climatologies, currently set to 15, i.e. 7 days each side of the target day of the year. 
    - `climatology` (list): The climatological period, currently `[1991, 2020]`
    - `quantiles` (list): The list of climatological quantiles to calculate, currently `[0.1, 0.25, 0.5, 0.75, 0.9]`
    - `ipath` (str): The path where to find the OISST archive (`domain` will be added to the root path)
    - `opath` (str): The path where the climatologies will be saved (in zarr format), `domain` will be added to the root path. 

Below are examples of the maps generated

<img src="https://github.com/nicolasfauchereau/OISST_indices/blob/main/figures/combo_images_anoms.png" alt="SST anomalies with MHW conditions stippled" title="SST anomalies with MHW conditions stippled" width="900"/>

<img src="https://github.com/nicolasfauchereau/OISST_indices/blob/main/figures/combo_images_heatwave_days.png" alt="Number of consecutive periods with heatwave conditions" title="Number of consecutive periods with heatwave conditions" width="900"/>

<hr>

The map below shows the geometries used for the definition of the 6 NZ coastal regions (["NNI", "WNI", "ENI", "NSI", "WSI", "ESI"])

<img src="https://github.com/nicolasfauchereau/OISST_indices/blob/main/figures/NZ_6_coastal_regions.png" alt="NZ 6 coastal regions" title="NZ 6 coastal regions" width="400"/>

and the following figure shows an example of the time-series generated 

<img src="https://github.com/nicolasfauchereau/OISST_indices/blob/main/figures/prototype_NZ_coastal_indices_1days_agg_to_20230227.png" alt="NZ 6 coastal regions time-series" title="NZ 6 coastal regions time-series" width="900"/>