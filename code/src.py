from email.mime import base


def preprocess(dset, domain):

    dset = dset.sel(lon=slice(*domain[:2]), lat=slice(*domain[2:]))

    return dset["sst"]


def timeout(timeout_secs: int):
    import time
    import signal
    from functools import wraps

    def wrapper(func):
        @wraps(func)
        def time_limited(*args, **kwargs):
            # Register an handler for the timeout
            def handler(signum, frame):
                raise Exception(f"Timeout for function '{func.__name__}'")

            # Register the signal function handler
            signal.signal(signal.SIGALRM, handler)

            # Define a timeout for your function
            signal.alarm(timeout_secs)

            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                raise exc
            finally:
                # disable the signal alarm
                signal.alarm(0)

            return result

        return time_limited

    return wrapper


def download_http(
    url=None,
    domain=[162.0, 180.0, -50.0, -30.0],
    opath=None,
):

    import requests
    import io
    import xarray as xr

    with requests.get(url) as r:

        dset = xr.open_dataset(io.BytesIO(r.content))

        dset = preprocess(dset, domain=domain)

        dset.to_netcdf(opath)

        dset.close()

    if opath.exists():

        return str(opath)

        print(
            f"{str(opath.joinpath(fname))} downloaded successfully using HTTPS PSL file server"
        )

    else:

        return None


@timeout(120)
def download_dap(
    url=None,
    domain=[162.0, 180.0, -50.0, -30.0],
    opath=None,
):

    import xarray as xr

    dset = xr.open_dataset(url)

    dset = preprocess(dset, domain)

    dset.to_netcdf(opath)

    dset.close()

    if opath.exists():

        return str(opath)

        print(
            f"{str(opath)} downloaded successfully using PSL THREDDS server"
        )

    else:

        return None


def download_OISST(
    dap_url="https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres",
    fileserver_url="https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres",
    year=None,
    domain=[162.0, 180.0, -50.0, -30.0],
    opath=None,
    verbose=True,
):

    import pathlib
    from datetime import datetime

    if not year:

        year = datetime.utcnow().year

    if not opath:

        opath = pathlib.Path.cwd().parent.joinpath("data")
        opath.mkdir(exist_ok=True)

    fname = f"sst.day.mean.{year}.v2.nc"

    try: 

        r = download_dap(url=f"{dap_url}/{fname}", domain=domain, opath=opath.joinpath(fname))
    
    except: 

        r = download_http(url=f"{fileserver_url}/{fname}", domain=domain, opath=opath.joinpath(fname))

def get_domain(name="NZ"):

    domains = {}
    domains["NZ"] = [162.0, 180.0, -50.0, -30.0]
    domains["Tropical_Pacific"] = [140.0, (360.0 - 70), -7, 7]
    domains["Ninos"] = [190.0, 280.0, -10.0, 5.0]
    domains["IOD"] = [45.0, 110.0, -10.0, 10.0]

    return domains[name]


def calculates_ninos(
    dset, lon_name="lon", lat_name="lat", nino="3.4", expand_dims=True
):

    import xarray as xr

    ninos = {
        "1+2": [270, 280, -10, 0],
        "3": [210, 270, -5, 5],
        "4": [160, 210, -5, 5],
        "3.4": [190, 240, -5, 5],
        "oni": [190, 240, -5, 5],
    }

    if nino == "all":

        l_ninos = []

        for nino_name in ninos.keys():

            sub = dset.sel(
                {
                    lon_name: slice(*ninos[nino_name][:2]),
                    lat_name: slice(*ninos[nino_name][2:]),
                }
            ).mean(dim=[lon_name, lat_name])

            sub = sub.expand_dims({"nino": [nino_name]})

            l_ninos.append(sub)

        sub = xr.concat(l_ninos, dim="nino")

    else:

        sub = dset.sel(
            {lon_name: slice(*ninos[nino][:2]), lat_name: slice(*ninos[nino][2:])}
        ).mean(dim=[lon_name, lat_name])

        if expand_dims:

            sub = sub.expand_dims({"nino": [nino]})

    return sub


def calculates_IOD_nodes(
    dset, lon_name="lon", lat_name="lat", IOD_node="IOD_West", expand_dims=True
):

    iod = {"IOD_West": [50, 70, -10, 10], "IOD_East": [90, 110, -10, 0]}

    if IOD_node == "all":

        l_iod = []

        for iod_name in iod.keys():

            sub = dset.sel(
                {
                    lon_name: slice(*iod[iod_name][:2]),
                    lat_name: slice(*iod[iod_name][2:]),
                }
            ).mean(dim=[lon_name, lat_name])

            sub = sub.expand_dims({"IOD": [iod_name]})

            l_iod.append(sub)

        sub = xr.concat(l_iod, dim="IOD")

    else:

        sub = dset.sel(
            {lon_name: slice(*iod[IOD_node][:2]), lat_name: slice(*iod[IOD_node][2:])}
        ).mean(dim=[lon_name, lat_name])

        if expand_dims:

            sub = sub.expand_dims({"IOD": [IOD_node]})

    return sub


def gpd_from_domain(lonmin=None, lonmax=None, latmin=None, latmax=None, crs="4326"):
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

    shape = Polygon(
        (
            (lonmin, latmin),
            (lonmax, latmin),
            (lonmax, latmax),
            (lonmin, latmax),
            (lonmin, latmin),
        )
    )

    shape_gpd = gpd.GeoDataFrame([], geometry=[shape])

    # set the CRS

    shape_gpd = shape_gpd.set_crs(f"EPSG:{crs}")

    return shape_gpd


def make_mask_from_shape(
    shapefile, dset, to_crs=None, lon_name="lon", lat_name="lat", mask_name="mask"
):

    import numpy as np
    import geopandas as gpd
    import regionmask

    shapefile = gpd.read_file(shapefile)

    if to_crs:

        shapefile = shapefile.to_crs(to_crs)

    lat = dset[lat_name].data
    lon = dset[lon_name].data

    mask = regionmask.mask_geopandas(shapefile, lon, lat)

    mask = mask.where(np.isnan(mask), other=1)

    mask.name = mask_name

    return mask


def fix_leapyears(dset):

    import numpy as np

    dset = dset.sel(time=~((dset.time.dt.month == 2) & (dset.time.dt.day == 29)))

    doy = np.arange(1, 366)

    doy = np.tile(doy, (len(dset.time) // 365) + 1)

    doy = doy[: len(dset.time)]

    dset["doy"] = (("time"), doy)

    return dset


def _interpolate_NaN(data):
    """ """

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


def interpolate_NaN_da(dataarray, lon_name="lon", lat_name="lat"):

    import xarray as xr

    regridded = xr.apply_ufunc(
        _interpolate_NaN,
        dataarray,
        input_core_dims=[[lat_name, lon_name]],
        output_core_dims=[[lat_name, lon_name]],
        vectorize=True,
        dask="allowed",
    )

    return regridded


def plot_SST_map(
    dataarray,
    mask,
    kind="anomalies",
    ndays_agg=1,
    cmap=None,
    stipples_color="k",
    levels=20,
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

    if kind == "anomalies":

        title = f"OISST V2 {ndays_agg} days average anomalies* to {date:%Y-%m-%d}\n"

        cbar_label = "SST anomaly (°C)"

    dataarray.plot.contourf(
        ax=ax,
        levels=levels,
        transform=ccrs.PlateCarree(),
        extend="both",
        cbar_kwargs={"shrink": 0.8, "pad": 0.05, "label": cbar_label},
        cmap=cmap,
    )

    cs = (dataarray * mask).plot.contourf(
        ax=ax,
        levels=levels,
        transform=ccrs.PlateCarree(),
        colors="None",
        hatches=[".."],
        add_colorbar=False,
    )

    for i, collection in enumerate(cs.collections):

        collection.set_edgecolor(stipples_color)
        collection.set_linewidth(0.0)

    ax.add_feature(land, facecolor="gainsboro")
    ax.add_feature(lakes, facecolor="b", edgecolor="b", alpha=0.2)
    ax.add_feature(states_provinces, edgecolor="k", linewidth=1)
    ax.coastlines("10m", linewidth=1)

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


from PIL import Image


def append_images(
    images, direction="horizontal", bg_color=(255, 255, 255), aligment="center"
):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == "horizontal":
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new("RGB", (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction == "horizontal":
            y = 0
            if aligment == "center":
                y = int((new_height - im.size[1]) / 2)
            elif aligment == "bottom":
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == "center":
                x = int((new_width - im.size[0]) / 2)
            elif aligment == "right":
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im
