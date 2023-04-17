#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import sys 
import pathlib
from datetime import date
import argparse

# %%
import OISST

# %% 
print(f"executing update_OISST.py with {str(sys.executable)}\n")

# %%
parser = argparse.ArgumentParser(prog = 'update_OISST.py',
                                description = 'Download the yearly OISST v2 netcdfs from the PSL', 
                                epilog = 'Always check data availability at https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/\n')

# %%
parser.add_argument('-y', '--year', type=int, default=None,
                    help="""The year to download the OISST v2 netcdf file for\n 
                    Note: - root URL is https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/\n
                          - The filename pattern is sst.day.mean.${year}.v2.nc""")
                    
parser.add_argument('-o', '--opath', type=str, default='/media/nicolasf/END19101/data/OISST/daily/',
                    help="The root path where to download the OISST datasets, default to '/media/nicolasf/END19101/data/OISST/daily/'")

parser.add_argument('-d', '--domain', type=str, default='NZ',
                    help="the domain for which to download / extract, can be in ['NZ','Ninos','IOD'], default to 'NZ'")
                
parser.add_argument('-t', '--tryDAP', type=int, default=0,
                    help="Whether to try the PSL DAP server, probably better to leave it at 0 (default, which means False)")
                    
args = parser.parse_args()

# %%
year = args.year 
opath = pathlib.Path(args.opath)
domain = args.domain
tryDAP = bool(args.tryDAP)

# %%
# year = 2022
# opath = pathlib.Path('/media/nicolasf/END19101/data/OISST/daily/') 
# domain = 'NZ'
# tryDAP = False

# %%
if year is None: 
    year = str(date.today().year)

# %%
download_path = opath.joinpath(domain)

# %%
domain = OISST.get_domain(domain)

# %%
OISST.download_OISST(year=year, opath=download_path, domain=domain, tryDAP=tryDAP)
