from utils_analysis import *

# Sentinel Hub configuration
config = get_config(
    sh_client_id="...",
    sh_client_secret="...",
    sh_base_url="https://creodias.sentinel-hub.com"
)


def dem_analysis(geometries, save_plots=False):
    """
    Analysis DEMs of given geometries.

    :param geometries: A dictionary containing the basins' names as keys and their corresponding geometries as values.
    :param save_plots: A boolean value indicating whether to save the generated elevation histograms. Default is False.
    :return: None
    """

    dem_dict = {}

    for basin_name, geometry in geometries.items():
        dem = get_data(
            evalscript_path="evalscripts/evalscript_dem.js",
            data_collection=DataCollection.DEM,
            geometry=geometry,
            time_interval=("2022-01-01", "2022-01-01"),
            resolution=(60, 60),
            config=config
        )

        dem_dict[basin_name] = dem

    plot_elevation_histograms(dem_dict, save=save_plots)


def main_analysis(basin_name, start_date, end_date, geometries, analyse_scattering=False, plot=False, save_plots=False):
    """
    Performs the main analysis for specified basin and time range.

    :param basin_name: The name of the basin for analysis.
    :param start_date: The start date of the analysis in the format "YYYY-MM-DD".
    :param end_date: The end date of the analysis in the format "YYYY-MM-DD".
    :param geometries: A dictionary containing the geometries of the basins.
    :param analyse_scattering: Boolean value indicating whether to analyze scattering (default is False).
    :param plot: Boolean value indicating whether to plot the results (default is False).
    :param save_plots: Boolean value indicating whether to save the plots (default is False).
    :return: A DataFrame containing the results of the analysis.
    """

    # Get basin geometry
    geometry = geometries[basin_name]

    # Track time
    start_time = time.time()

    # Store results
    results = {
        "YYYY-MM": [],
        "Rep. index (%)": [],
        "SLE (m a.s.l.)": [],
        "Err. index (%)": [],
        "Snow cover (%)": [],
        "Snow covered below - north (%)": [],
        "Snow covered below - south (%)": [],
        "Snow free above - north (%)": [],
        "Snow free above - south (%)": [],
        "Mean snow depth (cm)": [],
        "Mean SWE (mm)": []
    }

    # Get DEM
    dem = get_data(
        evalscript_path="evalscripts/evalscript_dem.js",
        data_collection=DataCollection.DEM,
        geometry=geometry,
        time_interval=(start_date, start_date),
        resolution=(60, 60),
        config=config
    )

    if start_date == end_date:
        # Get GFSC
        gfsc = get_data(
            evalscript_path="evalscripts/evalscript_gfsc.js",
            data_collection=DataCollection.define_byoc("e0e66010-ab8a-46d5-94bd-ae5c750e7341"),
            geometry=geometry,
            time_interval=(start_date, start_date),
            resolution=(60, 60),
            config=config
        )

        # Get Snow Depth
        snow_depth = get_data(
            "evalscripts/evalscript_snowdepth.js",
            data_collection=DataCollection.define_byoc("a34673e5-971d-4452-8fe7-7b2c43c7af4b"),
            geometry=geometry,
            time_interval=(start_date, start_date),
            resolution=(60, 60),
            config=config
        )

        # Get SWE
        swe = get_data(
            "evalscripts/evalscript_swe.js",
            data_collection=DataCollection.define_byoc("f7668b68-04aa-4e66-a164-21ae67991953"),
            geometry=geometry,
            time_interval=(start_date, start_date),
            resolution=(60, 60),
            config=config
        )

        # Get L1C
        l1c = get_data(
            evalscript_path="evalscripts/evalscript_l1c.js",
            data_collection=DataCollection.SENTINEL2_L1C,
            geometry=geometry,
            time_interval=(start_date, start_date),
            resolution=(60, 60),
            config=config
        )

        results["YYYY-MM"].append(start_date)

        analyse_snow(dem, gfsc, snow_depth, swe, results, analyse_scattering=analyse_scattering, plot=False,
                     save_plots=save_plots)

        snow_metrics = pd.DataFrame.from_dict(results)

        if plot:
            # Without SLE
            plot_aerial(dem, "DEM", save=save_plots)
            plot_aerial(gfsc, "GFSC", save=save_plots)
            plot_aerial(snow_depth, "SNOW DEPTH", save=save_plots)
            plot_aerial(swe, "SWE", save=save_plots)
            plot_aerial(l1c, "L1C", save=save_plots)

            # With SLE
            plot_sle_l1c_gfsc(l1c, gfsc, dem, [snow_metrics["SLE (m a.s.l.)"]], save=save_plots)
            plot_aerial(dem, "DEM", dem=dem, sle=[snow_metrics["SLE (m a.s.l.)"]], save=save_plots)
            plot_aerial(gfsc, "GFSC", dem=dem, sle=[snow_metrics["SLE (m a.s.l.)"]], save=save_plots)
            plot_aerial(l1c, "L1C", dem=dem, sle=[snow_metrics["SLE (m a.s.l.)"]], save=save_plots)

        print(f"{start_date} done! Runtime: {round((time.time() - start_time) / 60, 3)} minutes")

        return snow_metrics

    else:
        # Iterate through months
        for year_month in get_year_month_list(start_date, end_date):
            # Get first day, last day and name of month
            first_day, last_day = get_first_and_last_day_of_month(year_month)
            month_name = get_name_of_month(year_month)

            # Get GFSC (monthly aggregated)
            gfsc = get_data(
                evalscript_path="evalscripts/evalscript_gfsc_agg.js",
                data_collection=DataCollection.define_byoc("e0e66010-ab8a-46d5-94bd-ae5c750e7341"),
                geometry=geometry,
                time_interval=(first_day, last_day),
                resolution=(60, 60),
                config=config
            )

            # Get Snow Depth (monthly aggregated)
            snow_depth = get_data(
                "evalscripts/evalscript_snowdepth_agg.js",
                data_collection=DataCollection.define_byoc("a34673e5-971d-4452-8fe7-7b2c43c7af4b"),
                geometry=geometry,
                time_interval=(first_day, last_day),
                resolution=(60, 60),
                config=config
            )

            # Get SWE (monthly aggregated)
            swe = get_data(
                "evalscripts/evalscript_swe_agg.js",
                data_collection=DataCollection.define_byoc("f7668b68-04aa-4e66-a164-21ae67991953"),
                geometry=geometry,
                time_interval=(first_day, last_day),
                resolution=(60, 60),
                config=config
            )

            results["YYYY-MM"].append(year_month)

            analyse_snow(dem, gfsc, snow_depth, swe, results, analyse_scattering=analyse_scattering)

            print(
                f"{basin_name}: {month_name} {year_month[:4]} done! Runtime: {round((time.time() - start_time) / 60, 3)} minutes")

        snow_metrics = pd.DataFrame.from_dict(results)

        # Export
        filename = f"results/snow_analysis/{convert_to_snake_case(basin_name)}_{start_date}_{end_date}.csv"
        snow_metrics.to_csv(filename, index=False)


def time_series_analysis(basin_names_list, start_date, end_date, geometries, save_plots=False):
    """
    Performs time series analysis on snow metrics for the given basin names and time range.

    :param basin_names_list: List of basin names.
    :param start_date: Start date of the time series analysis.
    :param end_date: End date of the time series analysis.
    :param geometries: Dictionary of geometries for different basins.
    :param save_plots: Optional parameter to save plots. Defaults to False.
    :return: None
    """

    snow_metrics_dict = {}
    monthly_stats_dict = {}
    deviations_anomalies_dict = {}

    for basin_name in basin_names_list:
        filename = f"results/snow_analysis/{convert_to_snake_case(basin_name)}_{start_date}_{end_date}.csv"
        snow_metrics = pd.read_csv(filename)
        monthly_stats = get_monthly_stats(snow_metrics)
        deviations_anomalies = get_deviations_anomalies(snow_metrics, monthly_stats)

        snow_metrics_dict[basin_name] = snow_metrics
        monthly_stats_dict[basin_name] = monthly_stats
        deviations_anomalies_dict[basin_name] = deviations_anomalies

    # 0) Accuracy assessment
    plot_time_series(basin_names_list, snow_metrics_dict, "Rep. index (%)", save_plots)
    plot_time_series(basin_names_list, snow_metrics_dict, "Err. index (%)", save_plots)
    plot_scattering_time_series("Dora Baltea", snow_metrics_dict, "below", save_plots)
    plot_scattering_time_series("Dora Baltea", snow_metrics_dict, "above", save_plots)

    # 1) Time series
    plot_time_series(basin_names_list, snow_metrics_dict, "SLE (m a.s.l.)", save_plots)
    plot_time_series(basin_names_list, snow_metrics_dict, "Snow cover (%)", save_plots)
    plot_time_series(basin_names_list, snow_metrics_dict, "Mean snow depth (cm)", save_plots)
    plot_time_series(basin_names_list, snow_metrics_dict, "Mean SWE (mm)", save_plots)

    # 2) Deviation time series
    plot_time_series_deviation(basin_names_list, deviations_anomalies_dict, "SLE (m a.s.l.)", save_plots)

    # 3) Deviations by month
    plot_deviation_monthly(basin_names_list, deviations_anomalies_dict, "SLE (m a.s.l.)", "02", "deviation", save_plots)

    # 4) Deviations by basin
    # 2022
    timerange = ("2021-10", "2022-09")
    plot_deviation_bars(basin_names_list, deviations_anomalies_dict, "SLE (m a.s.l.)", timerange=timerange,
                        months=["12", "01", "02", "03", "04"], mode="deviation", save=save_plots)

    # 5) Aerial plot
    # Get L1C
    l1c = get_data(
        evalscript_path="evalscripts/evalscript_l1c.js",
        data_collection=DataCollection.SENTINEL2_L1C,
        geometry=geometries["Dora Baltea"],
        time_interval=("2022-02-13", "2022-02-13"),
        resolution=(60, 60),
        config=config
    )

    # Get DEM
    dem = get_data(
        evalscript_path="evalscripts/evalscript_dem.js",
        data_collection=DataCollection.DEM,
        geometry=geometries["Dora Baltea"],
        time_interval=("2022-02-13", "2022-02-13"),
        resolution=(60, 60),
        config=config
    )

    plot_aerial(l1c, "L1C", dem=dem, sle=[1789, 2134], save=save_plots)

    # 6) Time series comparison
    timerange = ("2018-10", "2022-09")
    for basin_name in basin_names_list:
        plot_time_series_comparison(basin_name, snow_metrics_dict, monthly_stats_dict, deviations_anomalies_dict,
                                    timerange=timerange, save=save_plots)

    # 7) Anomaly grid
    plot_grid(basin_names_list, deviations_anomalies_dict, ["SLE (m a.s.l.)", "Mean SWE (mm)"], "02", save=save_plots)


def gfsc_accuracy_assessment(test_areas_shapefile, date, save_plots=False):
    """
    Full workflow for assessing the accuracy of the GFSC against a custom snow classification.

    :param test_areas_shapefile: The file path to the shapefile containing the test areas.
    :param date: The date for which to assess the accuracy.
    :param save_plots: Boolean indicating whether to save the generated plots. Defaults to False if not provided.
    :return: None
    """

    # Store images
    gfsc_list = []
    custom_fsc_list = []

    # Store stats
    mae_list = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    # Get and plot test areas
    test_areas_geom_dict = get_test_areas(test_areas_shapefile, 3035)
    plot_test_areas(test_areas_shapefile, "results/imagery/dora_baltea/S2-L1C-20220414.tiff", save=save_plots)

    for geom_id, geom in test_areas_geom_dict.items():
        # Get L2A
        l2a = get_data(
            evalscript_path="evalscripts/evalscript_l2a.js",
            data_collection=DataCollection.SENTINEL2_L2A,
            geometry=geom,
            time_interval=(date, date),
            resolution=(10, 10),
            config=config,
        )

        # Get GFSC
        gfsc = get_data(
            evalscript_path="evalscripts/evalscript_gfsc.js",
            data_collection=DataCollection.define_byoc("e0e66010-ab8a-46d5-94bd-ae5c750e7341"),
            geometry=geom,
            time_interval=(date, date),
            resolution=(60, 60),
            config=config,
        )

        # Get custom snow cover
        custom_snow_cover = get_data(
            evalscript_path="evalscripts/evalscript_custom_snow_cover.js",
            data_collection=DataCollection.SENTINEL2_L2A,
            geometry=geom,
            time_interval=(date, date),
            resolution=(10, 10),
            config=config,
        )

        custom_fsc = downsample(custom_snow_cover, (6, 6))

        # Append images
        gfsc_list.append(gfsc)
        custom_fsc_list.append(custom_fsc)

        # Plots and similarity stats
        plot_l2a_custom_snow_cover(l2a, custom_snow_cover, geom_id, save=save_plots)
        plot_gfsc_custom_fsc(gfsc, custom_fsc, geom_id, save=save_plots)
        similarity_dict = plot_l2a_diff(l2a, gfsc, custom_fsc, geom_id, save=save_plots)

        mae_list.append(similarity_dict["MAE"])
        rmse_list.append(similarity_dict["RMSE"])
        psnr_list.append(similarity_dict["PSNR"])
        ssim_list.append(similarity_dict["SSIM"])

    for i in range(len(mae_list)):
        print(f"{mae_list[i]:.2f} & {rmse_list[i]:.2f} & {psnr_list[i]:.2f} & {ssim_list[i]:.2f}")

    plot_diff_grid(gfsc_list, custom_fsc_list, save=save_plots)


def snow_depth_analysis(basin_name, geometries, start_date, end_date):
    """
    Extracts the snow depth pixel values and compares them to in-situ snow depth measurements.

    :param basin_name: Name of the basin.
    :param geometries: Dictionary containing the geometries of the basins.
    :param start_date: Start date of the analysis (YYYY-MM-DD).
    :param end_date: End date of the analysis (YYYY-MM-DD).
    :return: None
    """

    # Get basin geometry
    geometry = geometries[basin_name]

    # Get dates
    date_list = get_date_list(start_date, end_date)

    # Stations
    stations_df = read_coordinates("data/meteo/stations.csv")
    transform = get_transform("results/imagery/dora_baltea/COP-DEM.tiff")
    transform_coordinates(stations_df, transform, 4326, 3035)

    # Read snow depth measurements
    ground_snow_depth = read_ground_snow_depth("data/meteo/snow_depth/")

    # Store results
    ground_snow_depth_df = pd.DataFrame(index=ground_snow_depth.index, columns=date_list)
    dta_snow_depth_df = pd.DataFrame(index=ground_snow_depth.index, columns=date_list)

    for date in date_list:
        # Get DTA snow depth
        dta_snow_depth_date = get_data(
            "evalscripts/evalscript_snowdepth.js",
            data_collection=DataCollection.define_byoc("a34673e5-971d-4452-8fe7-7b2c43c7af4b"),
            geometry=geometry,
            time_interval=(date, date),
            resolution=(60, 60),
            config=config
        )

        # Get ground snow depth
        ground_snow_depth_date = ground_snow_depth[date]

        # Get values for each station
        for i, station in stations_df.iterrows():
            name = station["Name"]
            row = int(station["row"])
            col = int(station["col"])

            ground_snow_depth_station = ground_snow_depth_date[name]
            dta_snow_depth_station = dta_snow_depth_date[row, col]

            # Store results
            ground_snow_depth_df.loc[name, date] = ground_snow_depth_station
            dta_snow_depth_df.loc[name, date] = dta_snow_depth_station

    # Write to csv
    stations_df.to_csv("results/snow_depth_validation/stations.csv", index=True)
    ground_snow_depth_df.to_csv("results/snow_depth_validation/ground_snow_depth.csv", index=True)
    dta_snow_depth_df.to_csv("results/snow_depth_validation/dta_snow_depth.csv", index=True)


def snow_depth_validation(basin_name, geometries, save_plots=False):
    """
    Validates the modelled snow depth against in-situ measurements.

    :param basin_name: The name of the basin for which snow depth validation is being performed.
    :param geometries: A dictionary containing geometries of different basins.
    :param save_plots: A boolean indicating whether to save the generated plots or not. Default is False.
    :return: None
    """

    # Read data
    stations_df = pd.read_csv("results/snow_depth_validation/stations.csv", index_col=0)
    ground_snow_depth_df = pd.read_csv("results/snow_depth_validation/ground_snow_depth.csv", index_col=0)
    dta_snow_depth_df = pd.read_csv("results/snow_depth_validation/dta_snow_depth.csv", index_col=0)

    # Create masked duplicate (same size for correlation required)
    dta_snow_depth_masked_df = dta_snow_depth_df.mask(ground_snow_depth_df.isna())

    # Store results
    ground_snow_depth_list = []
    dta_snow_depth_list = []

    # Station wise
    for i, station_name in enumerate(stations_df["Name"]):
        station_ground_snow_depth = ground_snow_depth_df.loc[station_name].dropna()
        station_snow_depth = dta_snow_depth_masked_df.loc[station_name].dropna()

        ground_snow_depth_list.append(station_ground_snow_depth)
        dta_snow_depth_list.append(station_snow_depth)

        # Sample size
        ss = station_ground_snow_depth.size

        # RMSE
        differences = station_snow_depth - station_ground_snow_depth
        rmse = np.sqrt(np.mean(differences ** 2))

        # Pearson's correlation coefficient
        correlation_coefficient = np.corrcoef(station_snow_depth, station_ground_snow_depth)
        r = correlation_coefficient[0, 1]

        # Store stats in dataframe
        stations_df.loc[i, "Sample size"] = ss
        stations_df.loc[i, "RMSE"] = rmse
        stations_df.loc[i, "Correlation"] = r

        # Plot time series and scatter plot
        plot_snow_depth_time_series(stations_df, ground_snow_depth_df, dta_snow_depth_df, station_name, save_plots)
        plot_snow_depth_scatter(stations_df, ground_snow_depth_list, dta_snow_depth_list, station_name, save=save_plots)

    # Overall agreement
    plot_snow_depth_scatter(stations_df, ground_snow_depth_list, dta_snow_depth_list,
                            mode="RMSE", save=save_plots)
    plot_snow_depth_scatter(stations_df, ground_snow_depth_list, dta_snow_depth_list,
                            mode="RMSE", drop_stations=True, save=save_plots)

    # Aerial plots
    geometry = geometries[basin_name]

    # Get L1C
    l1c = get_data(
        evalscript_path="evalscripts/evalscript_l1c.js",
        data_collection=DataCollection.SENTINEL2_L1C,
        geometry=geometry,
        time_interval=("2022-01-24", "2022-01-24"),
        resolution=(60, 60),
        config=config
    )

    # Plot weather stations on L1C
    plot_weather_stations(l1c, stations_df, save=save_plots)
    plot_weather_stations(l1c, stations_df, stations_color="RMSE", save=save_plots)
    plot_weather_stations(l1c, stations_df, stations_color="Correlation", save=save_plots)
