from plots import *


def get_basins(basins_shapefile, alpine_perimeter_shapefile, basin_names, crs):
    """
    Retrieves the geometry of basins based on the provided shapefiles, basin names, and coordinate reference system.

    :param basins_shapefile: Path to the shapefile containing the basins.
    :param alpine_perimeter_shapefile: Path to the shapefile containing the alpine perimeter.
    :param basin_names: List of basin names to retrieve.
    :param crs: Coordinate reference system (CRS) to convert the shapefiles to.
    :return: Dictionary mapping basin names to their corresponding geometries.
    """

    basins = gpd.read_file(basins_shapefile).to_crs(crs)
    alpine_perimeter = gpd.read_file(alpine_perimeter_shapefile).to_crs(crs)

    basins_geom_dict = {}

    for basin_name in basin_names:
        basin = basins[basins["SUB_NAME"] == basin_name].reset_index()
        basin = gpd.overlay(basin, alpine_perimeter, how="intersection").explode(index_parts=False)
        basin["area"] = basin["geometry"].area
        basin = basin[basin["area"] == basin["area"].max()].reset_index()
        basins_geom_dict[basin_name] = Geometry(basin.geometry[0], CRS(crs))

    return basins_geom_dict


def get_test_areas(test_areas_shapefile, crs):
    """
    Reads a shapefile containing test areas and converts them to the specified CRS.

    :param test_areas_shapefile: The path to the shapefile containing the test areas.
    :param crs: The coordinate reference system (CRS) to be used for the test areas.
    :return: A dictionary containing test area geometries with their respective CRS.
    """

    test_areas = gpd.read_file(test_areas_shapefile).to_crs(crs)

    test_areas_geom_dict = {}

    for i, row in test_areas.iterrows():
        test_areas_geom_dict[row.id] = Geometry(row.geometry, CRS(crs))

    return test_areas_geom_dict


def get_config(sh_client_id, sh_client_secret, sh_base_url):
    """
    Get Sentinel Hub configuration.

    :param sh_client_id: The client ID for the Sentinel Hub API authentication.
    :param sh_client_secret: The client secret for the Sentinel Hub API authentication.
    :param sh_base_url: The base URL for the Sentinel Hub API requests.
    :return: The SHConfig object containing the provided configuration values.
    """
    config = SHConfig()
    config.sh_client_id = sh_client_id
    config.sh_client_secret = sh_client_secret
    config.sh_base_url = sh_base_url
    return config


def get_data(evalscript_path, data_collection, geometry, time_interval, resolution, config, save_data=False):
    """
    Retrieves satellite imagery from Sentinel Hub.

    :param evalscript_path: The file path of the Evalscript to use for the Sentinel Hub request.
    :param data_collection: The name of the data collection to request data from.
    :param geometry: The geometry (bounding box or geometry object) to define the spatial extent of the request.
    :param time_interval: The time interval to retrieve data for. Format: ('yyyy-mm-dd', 'yyyy-mm-dd').
    :param resolution: The resolution of the requested data in meters.
    :param config: The configuration object for the Sentinel Hub request.
    :param save_data: A flag indicating whether to save the retrieved data to disk. Default is False.
    :return: The requested data as a numpy array.
    """

    with open(evalscript_path, "r") as file:
        evalscript = file.read()

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=data_collection,
                                                  time_interval=time_interval)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        geometry=geometry,
        resolution=resolution,
        config=config,
        data_folder="results/imagery"
    )

    return request.get_data(save_data=save_data)[0]


def estimate_sle(gfsc, dem, plot=False, save_plots=False):
    """
    Estimates SLE and related values using GFSC and DEM.

    :param gfsc: Gap-filled Fractional Snow Cover (GFSC) data as a numpy array.
    :param dem: Digital Elevation Model (DEM) data as a numpy array.
    :param plot: Optional parameter to indicate whether to plot the snow distribution. Default is False.
    :param save_plots: Optional parameter to indicate whether to save the plotted snow distribution. Default is False.
    :return: Dictionary containing the following information:
        - "Rep. index (%)": Representativeness index in percentage.
        - "SLE (m a.s.l.)": Snow line elevation (SLE) in meters above sea level.
        - "Err. index (%)": Error index in percentage.
        - "Snow cover (%)": Snow cover percentage.
    """

    # Calculate representativeness index RI
    total_pixels = np.count_nonzero(~np.isnan(dem))
    valid_pixels = np.count_nonzero(~np.isnan(gfsc))
    ri = (valid_pixels / total_pixels) * 100

    # Fractional snow cover and complement
    snow_covered = gfsc
    snow_free = 100 - snow_covered

    # Store results
    elev_list = []
    snow_covered_pixels_below_list = []
    snow_free_pixels_above_list = []

    # Get min and max elevation
    elev_min = np.nanmin(dem)
    elev_max = np.nanmax(dem)

    # Starting elevation
    elev = elev_min

    # Incrementation
    while elev <= elev_max:
        elev_list.append(elev)

        # Mask elevation below and above
        elev_mask_below = np.where(dem < elev, 1, np.nan).astype("float32")
        elev_mask_above = np.where(dem >= elev, 1, np.nan).astype("float32")

        # Mask snow below and land above
        snow_covered_below = snow_covered * elev_mask_below
        snow_free_above = snow_free * elev_mask_above

        # Count snow covered area below and above (km2)
        snow_covered_pixels_below = (np.nansum(snow_covered_below) / 100)
        snow_free_pixels_above = (np.nansum(snow_free_above) / 100)

        # Store results
        snow_covered_pixels_below_list.append(snow_covered_pixels_below)
        snow_free_pixels_above_list.append(snow_free_pixels_above)

        elev += 1

    snow_dist = pd.DataFrame(list(zip(elev_list, snow_covered_pixels_below_list, snow_free_pixels_above_list)),
                             columns=["Elevation (m)", "Snow covered below (px)", "Snow free above (px)"])

    # Get SLE
    sle_idx = np.abs(snow_dist["Snow covered below (px)"] - snow_dist["Snow free above (px)"]).idxmin()
    sle = snow_dist.loc[sle_idx, "Elevation (m)"]

    # Calculate error index EI
    err_area = snow_dist.loc[sle_idx, "Snow covered below (px)"] + snow_dist.loc[sle_idx, "Snow free above (px)"]
    ei = (err_area / total_pixels) * 100

    # Get area above SLE
    snow_cover = (np.nansum(np.where(dem >= sle, 1, 0).astype("float32")) / total_pixels) * 100

    # Plot distribution
    if plot:
        plot_snow_distribution(snow_dist, sle, save=save_plots)

    return {"Rep. index (%)": ri, "SLE (m a.s.l.)": sle, "Err. index (%)": ei, "Snow cover (%)": snow_cover}


def analyse_sle_scattering(dem, gfsc, sle):
    """
    Analyses the scattering of erroneous pixels by aspect.

    :param dem: The digital elevation model.
    :param gfsc: The fractional snow cover.
    :param sle: The snow line elevation.
    :return: A dictionary containing the percentages of snow cover and snow-free areas below and above the snow line in north and south directions.
    """

    # Fractional snow cover and complement
    snow_covered = gfsc
    snow_free = 100 - snow_covered

    # Mask elevation
    elev_mask_below = np.where(dem < sle, 1, np.nan).astype("float32")
    elev_mask_above = np.where(dem >= sle, 1, np.nan).astype("float32")

    # Mask snow below and snow free above
    snow_covered_below = snow_covered * elev_mask_below
    snow_free_above = snow_free * elev_mask_above

    # Mask aspect
    aspect = get_aspect(dem)
    north_mask = np.where((aspect > 270) | (aspect <= 90), 1, np.nan).astype("float32")
    south_mask = np.where((aspect > 90) & (aspect <= 270), 1, np.nan).astype("float32")

    # Snow covered below
    snow_covered_below_north = snow_covered_below * north_mask
    snow_covered_below_south = snow_covered_below * south_mask
    snow_covered_below_north_sum = np.nansum(snow_covered_below_north)
    snow_covered_below_south_sum = np.nansum(snow_covered_below_south)
    snow_covered_below_sum = snow_covered_below_north_sum + snow_covered_below_south_sum
    snow_covered_below_north_perc = (snow_covered_below_north_sum / snow_covered_below_sum) * 100
    snow_covered_below_south_perc = (snow_covered_below_south_sum / snow_covered_below_sum) * 100

    # Snow free above
    snow_free_above_north = snow_free_above * north_mask
    snow_free_above_south = snow_free_above * south_mask
    snow_free_above_north_sum = np.nansum(snow_free_above_north)
    snow_free_above_south_sum = np.nansum(snow_free_above_south)
    snow_free_above_sum = snow_free_above_north_sum + snow_free_above_south_sum
    snow_free_above_north_perc = (snow_free_above_north_sum / snow_free_above_sum) * 100
    snow_free_above_south_perc = (snow_free_above_south_sum / snow_free_above_sum) * 100

    sle_scattering = {"Snow covered below - north (%)": snow_covered_below_north_perc,
                      "Snow covered below - south (%)": snow_covered_below_south_perc,
                      "Snow free above - north (%)": snow_free_above_north_perc,
                      "Snow free above - south (%)": snow_free_above_south_perc}

    return sle_scattering


def analyse_snow_depth(snow_depth):
    """
    Returns mean snow depth.

    :param snow_depth: Numpy array containing snow depth in centimeters.
    :return: Dictionary containing the mean snow depth in centimeters.
    """

    if np.isnan(snow_depth).all():
        return {"Mean snow depth (cm)": np.nan}

    else:
        mean_snow_depth = np.nanmean(snow_depth)
        return {"Mean snow depth (cm)": mean_snow_depth}


def analyse_swe(swe):
    """
    Returns mean SWE.

    :param swe: Numpy array containing SWE in millimeters.
    :return: Dictionary containing the mean SWE in millimeters.
    """

    if np.isnan(swe).all():
        return {"Mean SWE (mm)": np.nan}
    else:
        mean_swe = np.nanmean(swe)
        return {"Mean SWE (mm)": mean_swe}


def analyse_snow(dem, gfsc, snow_depth, swe, results, analyse_scattering=False, plot=False, save_plots=False):
    """
    Estimates SLE and retrieves snow cover, mean snow depth and mean SWE from satellite imagery.

    :param dem: The digital elevation model (DEM) data.
    :param gfsc: The Gap-filled Fractional Snow Cover (GFSC) data.
    :param snow_depth: The snow depth data.
    :param swe: The snow water equivalent (SWE) data.
    :param results: The dictionary to store the results.
    :param analyse_scattering: Optional parameter to indicate whether to analyse the scattering by aspect. Default is False.
    :param plot: Optional parameter to indicate whether to plot the results. Default is False.
    :param save_plots: Optional parameter to indicate whether to save the plots. Default is False.
    :return: None
    """

    # Compute SLE
    sle_results = estimate_sle(gfsc, dem, plot=plot, save_plots=save_plots)
    ri = sle_results["Rep. index (%)"]
    sle = sle_results["SLE (m a.s.l.)"]
    ei = sle_results["Err. index (%)"]
    snow_cover = sle_results["Snow cover (%)"]

    # Analyse scattering by aspect
    if analyse_scattering:
        scattering_by_aspect = analyse_sle_scattering(dem, gfsc, sle)
        sc_below_north = scattering_by_aspect["Snow covered below - north (%)"]
        sc_below_south = scattering_by_aspect["Snow covered below - south (%)"]
        sf_above_north = scattering_by_aspect["Snow free above - north (%)"]
        sf_above_south = scattering_by_aspect["Snow free above - south (%)"]
    else:
        sc_below_north = np.nan
        sc_below_south = np.nan
        sf_above_north = np.nan
        sf_above_south = np.nan

    # Analyze snow depth
    snow_depth_results = analyse_snow_depth(snow_depth)
    mean_snow_depth = snow_depth_results["Mean snow depth (cm)"]

    # Analyse SWE
    swe_results = analyse_swe(swe)
    mean_swe = swe_results["Mean SWE (mm)"]

    # Add results
    results["Rep. index (%)"].append(ri)
    results["SLE (m a.s.l.)"].append(sle)
    results["Err. index (%)"].append(ei)
    results["Snow cover (%)"].append(snow_cover)
    results["Snow covered below - north (%)"].append(sc_below_north)
    results["Snow covered below - south (%)"].append(sc_below_south)
    results["Snow free above - north (%)"].append(sf_above_north)
    results["Snow free above - south (%)"].append(sf_above_south)
    results["Mean snow depth (cm)"].append(mean_snow_depth)
    results["Mean SWE (mm)"].append(mean_swe)


def get_monthly_stats(snow_metrics, timerange=("2016-10", "2022-09")):
    """
    Calculates various statistics for each snow metric on a monthly basis within the specified time range.

    :param snow_metrics: A pandas DataFrame containing snow metrics data.
    :param timerange: A tuple specifying the start and end dates for the time range of interest.
    :return: A dictionary containing monthly statistics for each snow metric in the given time range.
    """

    idx_start = snow_metrics[snow_metrics["YYYY-MM"] == timerange[0]].index.values[0]
    idx_end = snow_metrics[snow_metrics["YYYY-MM"] == timerange[1]].index.values[0]
    snow_metrics_valid = snow_metrics.iloc[idx_start:idx_end + 1]

    indicators = ["SLE (m a.s.l.)", "Snow cover (%)", "Mean snow depth (cm)", "Mean SWE (mm)"]
    monthly_stats = {}

    for indicator in indicators:
        monthly_stats[indicator] = snow_metrics_valid.groupby(
            snow_metrics_valid["YYYY-MM"].apply(lambda x: pl.parse(x).format("MM")), as_index=True, sort=False).agg(
            MIN=(indicator, "min"),
            MAX=(indicator, "max"),
            MEAN=(indicator, "mean"),
            MEDIAN=(indicator, "median"),
            STD=(indicator, "std"),
            P10=(indicator, lambda x: x.quantile(0.1)),
            P90=(indicator, lambda x: x.quantile(0.9))
        )

    return monthly_stats


def get_deviations_anomalies(snow_metrics, monthly_stats):
    """
    Calculates the deviations and anomalies for each indicator based on monthly stats.

    :param snow_metrics: DataFrame containing snow metrics data.
    :param monthly_stats: DataFrame containing monthly statistics data.
    :return: Dictionary containing DataFrames with deviations and anomalies for each indicator.
    """
    year_month_series = snow_metrics["YYYY-MM"]

    indicators = ["SLE (m a.s.l.)", "Snow cover (%)", "Mean snow depth (cm)", "Mean SWE (mm)"]
    results = {}

    for indicator in indicators:
        monthly_stats_indicator = monthly_stats[indicator]

        year_month_list = []
        deviation_list = []
        anomaly_list = []

        for i, year_month in enumerate(year_month_series):
            month = year_month[-2:]

            monthly_min = monthly_stats_indicator.loc[month, "MIN"]
            monthly_max = monthly_stats_indicator.loc[month, "MAX"]
            monthly_median = monthly_stats_indicator.loc[month, "MEDIAN"]

            snow_metrics_month = snow_metrics.iloc[i]
            snow_metrics_month_indicator = snow_metrics_month[indicator]

            deviation = snow_metrics_month_indicator - monthly_median
            anomaly = ((snow_metrics_month_indicator - monthly_min) / (monthly_max - monthly_min)) * 200 - 100

            year_month_list.append(year_month)
            deviation_list.append(deviation)
            anomaly_list.append(anomaly)

        results[indicator] = pd.DataFrame({"YYYY-MM": year_month_list,
                                           "deviation": deviation_list,
                                           "anomaly": anomaly_list})

    return results


def read_ground_snow_depth(path):
    """
    Reads the ground snow depth data from multiple files in the specified path.

    :param path: The path to the directory containing the files.
    :return: A DataFrame containing the ground snow depth data, sorted by station name.
    """

    snow_depth_dict = {}
    names = ["Date", "Snow depth [cm]"]

    for file in os.listdir(path):
        file_path = path + file
        header = pd.read_csv(file_path, skiprows=2, nrows=5, encoding="windows-1252", header=None)
        station = header.loc[0, 0][10:]

        data = pd.read_csv(file_path, encoding="windows-1252", skiprows=8, sep=";", header=None, names=names)

        data["Date"] = data["Date"].apply(lambda x: pl.parse(x).format("YYYY-MM-DD"))
        data["Snow depth [cm]"] = data["Snow depth [cm]"].str.replace(",", ".").astype("float")

        snow_depth = data["Snow depth [cm]"].values

        snow_depth_dict[station] = snow_depth

    snow_depth_dict = dict(sorted(snow_depth_dict.items(), key=lambda x: x[0].lower()))

    snow_depth_df = pd.DataFrame.from_dict(snow_depth_dict, orient="index", columns=data["Date"])

    return snow_depth_df


def read_coordinates(path):
    """
    Reads coordinates from a CSV file.

    :param path: The path to the CSV file.
    :return: A DataFrame with columns ["Name", "Elevation", "Latitude", "Longitude"].
    """

    data = pd.read_csv(path, sep=";", header=None, names=["Name", "Elevation", "Latitude", "Longitude"])
    return data


def get_transform(img_path):
    """
    Gets the transformation matrix of an image.

    :param img_path: The path to the image file.
    :return: The transformation matrix.
    """

    with rasterio.open(img_path, "r") as img:
        transform = img.transform

    return transform


def transform_coordinates(data, affine, src_epsg, dst_epsg):
    """
    Transforms coordinates from one CRS (Coordinate Reference System) to another.

    :param data: A pandas DataFrame containing the coordinates to transform.
    :param affine: A rasterio Affine object representing the affine transformation matrix.
    :param src_epsg: The EPSG code of the source CRS.
    :param dst_epsg: The EPSG code of the destination CRS.
    :return: The transformed coordinates are added to the input DataFrame as columns "X", "Y", "row", and "col".
    """

    src_crs = rasterio.crs.CRS.from_epsg(src_epsg)
    dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)

    for i, row in data.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        x, y = warp.transform(src_crs, dst_crs, [lon], [lat])
        row, col = rasterio.transform.rowcol(affine, x, y)
        data.loc[i, "X"] = x
        data.loc[i, "Y"] = y
        data.loc[i, "row"] = row
        data.loc[i, "col"] = col
