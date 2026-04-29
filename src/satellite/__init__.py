
import datetime as dt
import numpy as np
import numpy.typing as npt
from goes2go import GOES
from xarray import DataArray, Dataset
import s3fs


def generate_timestamps(
    start: dt.datetime = dt.datetime(2017, 3, 1, 0, 3, tzinfo=dt.UTC),
    end: dt.datetime = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
) -> list[dt.datetime]:
    """
    Generates a list of 5 minutes seperated datetimes starting on minute 3
    of each year 2018-2024 and 2017 without Jan and Feb

    Parameters
    ----------
    start: dt.datetime
        The beginning of the time range to fetch
    end: dt.datetime
        The end of the time range to fetch
    
    Returns
    -------
    a list of datetimes in the range
    """

    timestamps = [[] for _ in range(12)]
    current_time = start

    while current_time < end:
        timestamps[current_time.minute // 5].append(current_time)
        current_time = current_time + dt.timedelta(minutes=5)

    return timestamps

def fetch(timestamp: dt.datetime, satellite: GOES) -> Dataset:
    """
    Gets a single satellite image nearest to timestamp

    Parameters
    ----------
    timestamp: dt.datetime
        The timestamp for the time we want an image for.
    satellite: GOES
        GOES object representing the satellite we want to fetch from
    
    Returns
    -------
    :Dataset
        xarray dataset containing a satellite image.
    """
    return satellite.nearesttime(
        timestamp.replace(tzinfo=None),
        return_as="xarray",
        download=False,
        verbose=False
    )


def fetch_range(start: dt.datetime, end: dt.datetime, satellite: GOES) -> Dataset:
    """
    Fetches all images in range [start, end]

    Parameters
    ----------
    start: dt.datetime
        The starting timestamp for the time we want an image for.
    end: dt.datetime
        The ending timestamp for the time we want an image for.
    satellite: GOES
        GOES object representing the satellite we want to fetch from
    
    Returns
    -------
    :Dataset
        xarray dataset containing all satellite images in range.

    """
    return satellite.timerange(start, end, return_as="xarray", download=False)


def fetch_bands(data: Dataset, bands: list[int]) -> DataArray:
    """
    Extracts bands of interest from data retrieved from satellite.

    Parameters
    ----------
    data: Dataset
        xarray dataset containing a satellite image.
    bands: list[int]
        Bands in data we want to keep.
    
    Returns
    -------
    :DataArray
        The input data but only keeping bands listed in bands
    """
    da = (
        data[[f"CMI_C{band:02d}" for band in bands]]
        .to_dataarray(dim="band")
    )
    # Single images don't have a "t" dimension; add one so downstream code is consistent
    if "t" not in da.dims:
        da = da.expand_dims("t")
    return da.transpose("t", "y", "x", "band")


def radiance_to_brightness_temp(rad, fk1, fk2, bc1, bc2):
    """
    Convert radiance to brightness temperature using Planck function constants
    from the L1b file metadata.

    Parameters
    ----------
    rad: array - radiance values
    fk1, fk2, bc1, bc2: float - Planck function constants from the file

    Returns
    -------
    array of brightness temperatures in Kelvin
    """
    return (fk2 / np.log((fk1 / rad) + 1) - bc1) / bc2


# Make satellite number configurable based on date
def get_satellite_bucket(timestamp):
    # GOES-18 became primary January 2023, GOES-16 data continues but GOES-18 is preferred
    # For 2025 data, use GOES-18
    if timestamp.year >= 2023:
        return "noaa-goes18"
    return "noaa-goes16"


def fetch_l1b_bands(timestamp: dt.datetime, bands: list[int]) -> tuple:
    """
    Fetch individual L1b-RadC band files for GOES-16 CONUS and convert
    radiance to brightness temperature. Much faster than downloading the
    full CMI product since each band file is ~5-10 MB vs ~50-100 MB for CMI.

    Parameters
    ----------
    timestamp: dt.datetime
        Target time for satellite image
    bands: list[int]
        Band numbers to fetch (e.g., [8, 9, 10, 13, 14, 15])

    Returns
    -------
    tuple of (band_data, first_dataset)
        band_data: numpy array of shape (1, y, x, num_bands) with brightness temps
        first_dataset: xarray Dataset from the first band (for coordinate extraction)
    """
    import xarray as xr

    fs = s3fs.S3FileSystem(anon=True)
    ts = timestamp.replace(tzinfo=None)

    # Build S3 prefix for the target hour
    bucket = get_satellite_bucket(ts)
    prefix = f"{bucket}/ABI-L1b-RadC/{ts.year}/{ts.timetuple().tm_yday:03d}/{ts.hour:02d}"

    all_band_data = []
    first_ds = None

    for band in bands:
        band_str = f"C{band:02d}"
        # List files matching this band in this hour
        try:
            files = fs.ls(prefix)
        except Exception:
            files = []

        # Find the file closest to our timestamp for this band
        band_files = [f for f in files if band_str in f]
        if not band_files:
            # Try the previous hour
            prev_ts = ts - dt.timedelta(hours=1)
            prev_prefix = f"noaa-goes16/ABI-L1b-RadC/{prev_ts.year}/{prev_ts.timetuple().tm_yday:03d}/{prev_ts.hour:02d}"
            try:
                files = fs.ls(prev_prefix)
            except Exception:
                files = []
            band_files = [f for f in files if band_str in f]

        if not band_files:
            raise FileNotFoundError(f"No L1b-RadC file found for band {band} near {timestamp}")

        # Pick the file closest to our target time
        # Filenames contain timestamps like: ..._sYYYYDDDHHMMSSS_...
        best_file = None
        best_diff = float('inf')
        for f in band_files:
            try:
                # Extract start time from filename: _sYYYYDDDHHMMSSS_
                parts = f.split('_s')
                if len(parts) < 2:
                    continue
                time_str = parts[1][:14]  # YYYYDDDHHMMSS
                file_year = int(time_str[0:4])
                file_doy = int(time_str[4:7])
                file_hour = int(time_str[7:9])
                file_min = int(time_str[9:11])
                file_sec = int(time_str[11:13])
                file_dt = dt.datetime(file_year, 1, 1) + dt.timedelta(
                    days=file_doy - 1, hours=file_hour, minutes=file_min, seconds=file_sec
                )
                diff = abs((file_dt - ts).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_file = f
            except (ValueError, IndexError):
                continue

        if best_file is None:
            raise FileNotFoundError(f"Could not parse L1b files for band {band} near {timestamp}")

        # Open the file directly from S3, load into memory before closing
        with fs.open(best_file) as fobj:
            ds = xr.open_dataset(fobj, engine='h5netcdf')
            ds.load()  # Force all data into memory before file handle closes

        if first_ds is None:
            first_ds = ds

        # Extract radiance and convert to brightness temperature
        rad = ds['Rad'].values
        fk1 = float(ds['planck_fk1'].values)
        fk2 = float(ds['planck_fk2'].values)
        bc1 = float(ds['planck_bc1'].values)
        bc2 = float(ds['planck_bc2'].values)

        bt = radiance_to_brightness_temp(rad, fk1, fk2, bc1, bc2)
        all_band_data.append(bt)

    # Stack bands: each is (y, x) -> result is (1, y, x, num_bands)
    stacked = np.stack(all_band_data, axis=-1)[np.newaxis, ...]
    return stacked.astype(np.float32), first_ds


def calculate_coordinates(data: Dataset) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Takes a xarray dataset of a satellite image and returns the lat-lon data
    for each datapoint in the array.

    Code pulled from the NOAA Documentation for the AWS Bucket

    data: Dataset
        The input xarray dataset
    
    Returns
    -------
    :tuple[npt.ArrayLike, npt.ArrayLike]
        (Lon coords, Lat coords)
    """
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = data.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = data.variables["y"][:]  # N/S elevation angle in radians
    projection_info = data.variables["goes_imager_projection"]
    lon_origin = projection_info.attrs["longitude_of_projection_origin"]
    H = (
        projection_info.attrs["perspective_point_height"]
        + projection_info.attrs["semi_major_axis"]
    )
    r_eq = projection_info.attrs["semi_major_axis"]
    r_pol = projection_info.attrs["semi_minor_axis"]

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * (s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y)))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return abi_lat, abi_lon


def project(
    lat: npt.ArrayLike | npt.DTypeLike,
    lon: npt.ArrayLike | npt.DTypeLike,
    temps: npt.ArrayLike | npt.DTypeLike,
) -> npt.ArrayLike | npt.DTypeLike:
    """
    Projects satellite image onto our grid with equal spaced lat and lon on x and y axes

    Parameters
    ----------
    lat: npt.ArrayLike | npt.DTypeLike
        Latitude coordinates
    lat: npt.ArrayLike | npt.DTypeLike
        Longitude coordinates
    lat: npt.ArrayLike | npt.DTypeLike
        Temprature values
    
    Returns
    -------
    :npt.ArrayLike | npt.DTypeLike
        A grid representing CONUS with data projected into the proper location
    """
    from consts import GRID_RANGE, MAP_RANGE
    from utils.convert import convert_coord as convert

    # Create coordinate mask
    lat_mask = (lat >= MAP_RANGE["LAT"]["MIN"]) & (lat <= MAP_RANGE["LAT"]["MAX"])
    lon_mask = (lon >= MAP_RANGE["LON"]["MIN"]) & (lon <= MAP_RANGE["LON"]["MAX"])
    mask = lat_mask & lon_mask

    # Map to grid coordinates
    rows = convert(lat[mask], "LAT")
    cols = convert(lon[mask], "LON")
    temps = temps[:, mask]

    # Shape data into grid
    data = np.zeros(
        (temps.shape[0], GRID_RANGE["LAT"], GRID_RANGE["LON"], temps.shape[-1]),
        dtype=np.float32,
    )
    data[:, rows, cols, :] = temps

    return data


# Currently this smooths a single band in a single file *(MAY NOT WORK ON EVERY BAND)
# Expects 1500 x 2500
# want (F, 1500, 2500, b)
#   F - num files
#   B - num Bands
# Seems like when we get multiple bands we are smoothing across multiple bands :(
def smooth(all_data: npt.ArrayLike | npt.DTypeLike) -> npt.ArrayLike | npt.DTypeLike:
    """
    Fills in the gaps for all bands in the data. Due to projection, some pixels may
    not have values and need to be interpolated.

    all_data: npt.ArrayLike | npt.DTypeLike

    Returns
    -------
    :npt.ArrayLike | npt.DTypeLike
        Interpolated data.
    """
    n_files = all_data.shape[0]
    n_bands = all_data.shape[3]

    for f in range(n_files):
        for b in range(n_bands):
            all_data[f, :, :, b] = smooth_single_band(all_data[f, :, :, b])
    return all_data


def smooth_single_band(
    data: npt.ArrayLike | npt.DTypeLike,
) -> npt.ArrayLike | npt.DTypeLike:
    """
    Fills in the gaps for a single band of data

    Parameters
    ----------
    data: npt.ArrayLike
        A single band of pirep data
    
    Returns
    -------
    npt.ArrayLike
        The single band of pirep data smoothed
    """
    from scipy.ndimage import distance_transform_edt

    ## runs after projection, gets points that are in between buckets from
    #  projection AND things that are not inside frame of GOES E
    empty_mask = (data <= 0) | np.isnan(data)
    indices = distance_transform_edt(
        empty_mask, return_distances=False, return_indices=True
    )
    data[empty_mask] = data[tuple(indices[:, empty_mask])]
    return data

