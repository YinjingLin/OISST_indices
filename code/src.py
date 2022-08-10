def preprocess(dset, domain): 
    
    dset = dset.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))
    
    return dset['sst']

def download_OISST(base_url = 'https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres', year=None, domain=[162., 180., -50., -30.], opath=None, verbose=True):

    import pathlib
    import xarray as xr
    from datetime import datetime

    if not year: 

        year = datetime.utcnow().year
    
    if not opath:

        opath = pathlib.Path.cwd().parent.joinpath('data')
        opath.mkdir(exist_ok=True)

    fname = f"sst.day.mean.{year}.v2.nc" 

    url = f"{base_url}/{fname}"

    dset = xr.open_dataset(url)

    dset = preprocess(dset,domain)

    dset.to_netcdf(opath.joinpath(fname))

    dset.close()
     
    if verbose and opath.joinpath(fname).exists(): 

        print(f"{fname} successfully saved in {str(opath)}") 

def get_domain(name='NZ'):

    domains = {}
    domains['NZ'] = [162., 180., -50., -30.]
    domains['Tropical_Pacific'] = [140., (360. - 70), -7, 7]
    domains['Ninos'] = [190., 280., -10., 5.]
    domains['IOD'] = [45., 110., -10., 10.]

    return domains[name]

def calculates_ninos(dset, lon_name='lon', lat_name='lat', nino='3.4', expand_dims=True): 

    import xarray as xr

    ninos = {
        '1+2' : [270,280,-10,0],
        '3'   : [210,270,-5,5],
        '4'   : [160,210,-5,5],
        '3.4' : [190,240,-5,5],
        'oni' : [190,240,-5,5],
    }

    if nino == 'all': 
        
        l_ninos = []
        
        for nino_name in ninos.keys(): 
            
            sub = dset.sel({lon_name:slice(*ninos[nino_name][:2]),lat_name:slice(*ninos[nino_name][2:])}).mean(dim=[lon_name,lat_name])
            
            sub = sub.expand_dims({'nino':[nino_name]})
            
            l_ninos.append(sub)
            
        sub = xr.concat(l_ninos, dim='nino')
        
    else: 
    
        sub = dset.sel({lon_name:slice(*ninos[nino][:2]),lat_name:slice(*ninos[nino][2:])}).mean(dim=[lon_name,lat_name])
        
        if expand_dims: 
            
            sub = sub.expand_dims({'nino':[nino]})
            
    return sub

def calculates_IOD_nodes(dset, lon_name='lon', lat_name='lat', IOD_node='IOD_West', expand_dims=True): 

    iod = {
        'IOD_West' : [50,70,-10,10],
        'IOD_East'   : [90,110,-10,0]
    }

    if IOD_node == 'all': 
        
        l_iod = []
        
        for iod_name in iod.keys(): 
            
            sub = dset.sel({lon_name:slice(*iod[iod_name][:2]),lat_name:slice(*iod[iod_name][2:])}).mean(dim=[lon_name,lat_name])
            
            sub = sub.expand_dims({'IOD':[iod_name]})
            
            l_iod.append(sub)
            
        sub = xr.concat(l_iod, dim='IOD')
        
    else: 
    
        sub = dset.sel({lon_name:slice(*iod[IOD_node][:2]),lat_name:slice(*iod[IOD_node][2:])}).mean(dim=[lon_name,lat_name])
        
        if expand_dims: 
            
            sub = sub.expand_dims({'IOD':[IOD_node]})
            
    return sub

def gpd_from_domain(lonmin=None, lonmax=None, latmin=None, latmax=None, crs='4326'): 
    """
    creates a geopandas dataframe with a rectangular domain geometry from 
    min and max longitudes and latitudes
    can be called using gpd_from_domain(*[lonmin, lonmax, latmin, latmax])
    
    can be passed e.g. to get_one_GCM() or get_GCMs() as a `mask` keyword argument
 
    Parameters
    ----------
    lonmin : float, optional
        min longitude, by default None
    lonmax : float, optional
        max longitude, by default None
    latmin : float, optional
        min latitude, by default None
    latmax : float, optional
        max latitude, by default None
    crs : str, optional
        The coordinate reference system, by default '4326'
    Returns
    -------
    [type]
        [description]
    """

    from shapely.geometry import Polygon
    import geopandas as gpd 

    # make the box 
    
    shape = Polygon(((lonmin, latmin), (lonmax, latmin), (lonmax, latmax), (lonmin, latmax), (lonmin, latmin)))  
    
    shape_gpd = gpd.GeoDataFrame([], geometry=[shape])
    
    # set the CRS 
    
    shape_gpd = shape_gpd.set_crs(f'EPSG:{crs}')
    
    return shape_gpd

def fix_leapyears(dset): 

    import numpy as np
    
    dset = dset.sel(time=~((dset.time.dt.month == 2) & (dset.time.dt.day == 29)))
    
    doy = np.arange(1,366)
    
    doy = np.tile(doy, (len(dset.time) // 365) + 1) 
    
    doy = doy[:len(dset.time)]
    
    dset['doy'] = (('time'), doy)
    
    return dset
    
def _interpolate_NaN(data):
    """
    """   

    import numpy as np 
    from scipy import interpolate 
    
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    
    # mask invalid values
    array = np.ma.masked_invalid(data)
    
    # get grid
    xx, yy = np.meshgrid(x, y)
    
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    
    newarr = array[~array.mask]
    
    interp = interpolate.NearestNDInterpolator(list(zip(x1, y1)), newarr)
    
    x_out = x
    y_out = y
    
    xx, yy = np.meshgrid(x_out, y_out)
    
    return interp(xx, yy)

def interpolate_NaN_da(dataarray, lon_name='lon', lat_name='lat'): 

    import xarray as xr

    regridded = xr.apply_ufunc(_interpolate_NaN, dataarray,
                           input_core_dims=[[lat_name,lon_name]],
                           output_core_dims=[[lat_name,lon_name]],
                           vectorize=True, dask="allowed")
    
    return regridded

def plot_SST_map(
    dataarray, mask, kind="anomalies", ndays_agg=1, cmap=None, stipples_color="k"
):

    from datetime import datetime
    import numpy as np
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmocean

    if cmap is None:

        cmap = cmocean.cm.balance

    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )

    lakes = cfeature.NaturalEarthFeature("physical", "lakes", "10m")
    rivers = cfeature.NaturalEarthFeature("physical", "rivers_lake_centerlines", "10m")
    land = cfeature.NaturalEarthFeature("physical", "land", "10m")

    # get the min and max of the dataarray

    sst_range = np.round(dataarray.min(("lat", "lon")).data, 2), np.round(
        dataarray.max(("lat", "lon")).data, 2
    )

    date = datetime(
        *list(
            map(
                int,
                [
                    dataarray.time.dt.year,
                    dataarray.time.dt.month,
                    dataarray.time.dt.day,
                ],
            )
        )
    )

    f, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180.0)},
    )

    ax.coastlines("10m", linewidth=2)
    ax.add_feature(land, facecolor="gainsboro")
    ax.add_feature(lakes, facecolor="b", edgecolor="b", alpha=0.2)
    ax.add_feature(states_provinces, edgecolor="k", linewidth=1)

    if kind == "anomalies":

        title = f"OISST V2 {ndays_agg} days average anomalies* to {date:%Y-%m-%d}\n"

        cbar_label = "SST anomaly (°C)"

    dataarray.plot.contourf(
        ax=ax,
        levels=20,
        transform=ccrs.PlateCarree(),
        extend="both",
        cbar_kwargs={"shrink": 0.8, "pad": 0.05, "label": cbar_label},
        cmap=cmap,
    )

    cs = (dataarray * mask).plot.contourf(
        ax=ax,
        levels=20,
        transform=ccrs.PlateCarree(),
        colors="None",
        hatches=[".."],
        add_colorbar=False,
    )

    for i, collection in enumerate(cs.collections):

        collection.set_edgecolor(stipples_color)
        collection.set_linewidth(0.0)

    ax.set_title(None)

    ax.set_title(title, fontsize=14, loc="left", ha="left")
    
    ax.text(
        0.01,
        1.01,
        f"Max = {sst_range[1]:+4.2f}˚C | Min = {sst_range[0]:+4.2f}˚C",
        transform=ax.transAxes,
        ha="left",
        fontsize=12,
    )
    
    ax.text(
        0.01,
        -0.05,
        "*Marine heatwave conditions stippled",
        transform=ax.transAxes,
        ha="left",
        fontsize=12,
        color="k",
        style="italic",
    )

    return (f, ax)

