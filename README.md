## Sea Surface Temperature (SST) indices and maps derived from the daily OISST v2 product 

This repository contains notebooks and code (in code/src.py) to calculate, maps and plots a series 
of diagnostics 

The notebooks are intended to be run via [papermill](https://papermill.readthedocs.io/)

 1) **get_OISST_OpenDAP.ipynb** downloads an archive of the OISST V2 dataset from URL: [https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres](https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres). Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, currently can be ['NZ', 'Ninos', 'IOD']
    - `first_year` (int): The first year of the period to download, by default will download everything from this year to the current year
    - `opath` (str): The path where the OISST archive will be saved (in {`opath`}/{`domain`})        
  
<hr>

2) **update_OISST.ipynb** update the archive for all domains (i.e. download the current year's file). Parameters are: 

    - `domains` (list): The list of domain's datasets to update, currently ['NZ', 'Ninos', 'IOD']
    - `opath` (str): The path where the current year's file will be saved (in {`opath`}/{`domain`})          

<hr>

3) **calculate_OISST_climatologies.ipynb** calculates the climatological values for a particular domain, i.e. calculates average, std, and quantiles [0.1, 0.25, 0.5, 0.75, 0.9] for a given climatology period (currently 1991 - 2020). Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, currently can be ['NZ', 'Ninos', 'IOD']
    - `ndays` (int): The aggregation period (i.e. 1, 7 or 30 days)
    - `roll` (int): The buffer (in days) around the day of the year for the calculation of the climatologies, currently set to 15, i.e. 7 days each side of the target day of the year. 
    - `climatology` (list): The climatological period, currently `[1991, 2020]`
    - `quantiles` (list): The list of climatological quantiles to calculate, currently `[0.1, 0.25, 0.5, 0.75, 0.9]`
    - `ipath` (str): The path where to find the OISST archive (`domain` will be added to the root path)
    - `opath` (str): The path where the climatologies will be saved (in zarr format), `domain` will be added to the root path. 

*Note*: This notebook only needs to be run if calculating climatologies for another domain or another period.     

<hr>

4) **OISST_realtime_NZ_maps.ipynb** calculate and maps the SST anomalies around NZ, with heatwave conditions stippled, as well as calculate the number of consecutive periods (1, 7, or 30 days averages) to date where SSTs exceeded the 90th percentile. Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, currently can be ['NZ', 'Ninos', 'IOD']
    - `ipath` (str): The path where to find the OISST archive (`domain` will be added to the root path)
    - `clim_path` (str): The path where to find climatologies (`domain` will be added to the root path)
    - `fig_path` (str): The path where to save the figures
    - `ndays_agg` (int): The aggregation period (i.e. 1, 7 or 30 days)
    - `ndays_back` (int): The number of days to read
    - `quantile` (float): The quantile for the determination of heatwave conditions, default is 0.9 (90th percentile)
    - `lag` (int): The lag in days, currently set to zero: the last available period  

<hr>

5) **OISST_realtime_NZ_indices.ipynb** calculate and plots the NZ coast SST indices. Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, needs to be 'NZ' so do not change
    - `NZ_regions` (list): The list of NZ regions ["NNI", "WNI", "ENI", "NSI", "WSI", "ESI"]
    - `ipath` (str): The path where to find the OISST archive (`domain` will be added to the root path)
    - `clim_path` (str): The path where to find climatologies (`domain` will be added to the root path)
    - `shapes_path` (str): The path where to find the shapefiles (i.e. *{NZ_region}_buffered_50km.shp*)
    - `fig_path` (str): The path where to save the figures
    - `ndays_agg` (int): The aggregation period (i.e. 1, 7 or 30 days)
    - `nmonths_back` (int): How many past months to plot

<hr>

6) **OISST_realtime_Ninos_indices.ipynb** calculate and plots the Nino indices (1+2, 3, 4, 3.4). Parameters are: 

    - `domain` (str): The domain used by the function `get_domain`, needs to be 'Ninos' so do not change
    - `ipath` (str): The path where to find the OISST archive (`domain` will be added to the root path)
    - `clim_path` (str): The path where to find climatologies (`domain` will be added to the root path)
    - `fig_path` (str): The path where to save the figures
    - `ndays_agg` (int): The aggregation period (i.e. 1, 7 or 30 days)
    - `nmonths_back` (int): How many past months to plot








