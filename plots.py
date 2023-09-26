from utils import *

# Directory for saving figures
fig_dir = "..."

# General settings
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.labelpad"] = 10
plot_title = False


def plot_elevation_histograms(dem_dict, save=False):
    """
    Plots elevation histograms for each basin in the given DEM dictionary.

    :param dem_dict: A dictionary containing DEMs for different basins.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8.2, 5))
    colors = get_colors(len(dem_dict))

    for i, (basin_name, dem) in enumerate(dem_dict.items()):
        # Flatten and remove NaN values
        dem_flat = dem.flatten()
        dem_flat = dem_flat[~np.isnan(dem_flat)]

        # Define bins
        area = dem_flat.size * (3600 / 1000000)
        min_elev = np.min(dem_flat)
        max_elev = np.max(dem_flat)
        mean_elev = np.mean(dem_flat)

        elevation_bins = np.arange(int(min_elev / 100) * 100, int(max_elev / 100) * 100 + 200, 100)

        # Count pixels and pixel values
        hist, bins = np.histogram(dem_flat, bins=elevation_bins)

        ax.plot(hist, bins[:-1], label=basin_name, color=colors[i])

    # x-axis settings
    ax.set_xlabel("Frequency")
    ax.set_xticks([])

    # y-axis settings
    ax.set_ylim(-100, 5100)
    ax.set_ylabel("Elevation (m a.s.l.)")

    # Legend
    ax.legend(loc="upper right")

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)

    if save:
        plt.savefig(f"{fig_dir}/study_area/elevation_histograms", dpi=300)

    plt.show()


def plot_snow_distribution(snow_distribution, sle, save=False):
    """
    Plots the snow distribution based on the given dictionary.

    :param snow_distribution: A dictionary containing the snow distribution data.
    :param sle: A scalar value representing the SLE.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """
    elevation = snow_distribution["Elevation (m)"]
    snow_covered_below = snow_distribution["Snow covered below (px)"]
    snow_free_above = snow_distribution["Snow free above (px)"]

    # Figure settings
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = get_colors(3)

    # Distribution
    ax.plot(elevation, snow_covered_below, color=colors[0], label="Snow covered pixels below elev.")
    ax.plot(elevation, snow_free_above, color=colors[1], label="Snow free pixels above elev.")

    # SLE
    ax.axvline(x=sle, color="deeppink")
    ax.text(sle + 50, 450000, "SLE", ha="left", va="center", color="deeppink")

    ax.axvline(x=np.min(elevation), color=colors[2])
    ax.text(np.min(elevation) + 50, 50000, "Min. elev.", ha="left", va="center", color=colors[2])

    ax.axvline(x=np.max(elevation), color=colors[2])
    ax.text(np.max(elevation) - 50, 50000, "Max. elev.", ha="right", va="center", color=colors[2])

    # x-axis settings
    ax.set_xlabel("Elevation (m a.s.l.)")
    ax.set_xticks(np.arange(500, 4501, 500))

    # y-axis settings
    ax.set_ylabel("Pixels")
    ax.ticklabel_format(style="plain")

    # Legend
    ax.legend(loc="center right")

    # General
    ax.grid()
    remove_spines(ax)
    fig.tight_layout()

    if save:
        plt.savefig(f"{fig_dir}sle_estimation/snow_distribution", dpi=300)

    plt.show()


def plot_sle_l1c_gfsc(l1c, gfsc, dem, sle, save=False):
    """
    Plots the SLE as contour lines on Sentinel-2 L1C and HR-S&I GFSC images.

    :param l1c: 2D array representing the Sentinel-2 L1C image.
    :param gfsc: 2D array representing the HR-S&I GFSC image.
    :param dem: 2D array representing the Digital Elevation Model.
    :param sle: List of contour levels for the SLE.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """
    # Figure settings
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left plot
    ax[0].imshow(l1c)
    ax[0].contour(dem, levels=sle, colors="deeppink", zorder=50, linewidths=1.5)
    ax[0].set_title("Sentinel-2 L1C", fontsize="large", pad=10, loc="center")
    # ax[1].set_title(f"SLE = {int(sle[0])} m a.s.l.", fontsize="large", pad=10, loc="right")

    # Right plot
    gfsc = ax[1].imshow(gfsc, cmap="Blues", vmin=0, vmax=100)
    ax[1].contour(dem, levels=sle, colors="deeppink", zorder=50, linewidths=1.5)
    ax[1].set_title("HR-S&I GFSC", fontsize="large", pad=10, loc="center")

    remove_spines(ax[0], tick_labels_off=True)
    remove_spines(ax[1], tick_labels_off=True)
    plt.subplots_adjust(left=0.025, right=1.055, top=1, bottom=-0.05, wspace=0.1)

    # Colorbar
    cbar = fig.colorbar(gfsc, ax=ax, orientation="vertical", pad=0.04, shrink=0.6)
    cbar.ax.set_ylabel("Fractional snow cover (%)", labelpad=5)

    if save:
        plt.savefig(f"{fig_dir}sle_estimation/sle_aerial", dpi=300)

    plt.show()


def plot_aerial(image, image_type, dem=None, sle=None, plot_title=plot_title, save=False):
    """
    Plots aerial images of different types with or without SLE as contour line.

    :param image: The aerial image to plot.
    :param image_type: The type of the image. Possible values are "DEM", "GFSC", "SNOW DEPTH", "SWE", "SNOW COVERED", "SNOW FREE", and "L1C".
    :param dem: The digital elevation model (DEM) data. Required when image_type is "SNOW COVERED", "SNOW FREE", or "L1C" and sle is not None.
    :param sle: The snow line elevation (SLE) data. Can be a single value or a list of two values.
    :param plot_title: The title of the plot. Optional. Defaults to None.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.5))

    match image_type:
        case "DEM":
            cmap = truncate_colormap(plt.get_cmap("terrain"), 0.25, 1)
            cbar_title = "Elevation [m]"
            vmin = 0
            vmax = 5000
            title = "COPERNICUS DEM"
            filename = "dem"

            ls = LightSource(azdeg=30, altdeg=30)
            ax.imshow(ls.hillshade(image, vert_exag=0.01), cmap="gray", alpha=0.4, zorder=20)

        case "GFSC":
            cmap = "Blues"
            cbar_title = "Fractional Snow Cover [%]"
            vmin = 0
            vmax = 100
            title = "GFSC"
            filename = "gfsc"

        case "SNOW DEPTH":
            cmap = "Oranges"
            cbar_title = "Snow depth [cm]"
            vmin = 0
            vmax = 200
            title = "SNOW DEPTH"
            filename = "snow_depth"

        case "SWE":
            cmap = "Purples"
            cbar_title = "Snow Water Equivalent [mm]"
            vmin = 0
            vmax = 600
            title = "SWE"
            filename = "swe"

        case "SNOW COVERED":
            cmap = "Blues"
            cbar_title = "Snow covered [%]"
            vmin = 0
            vmax = 100
            title = "SNOW COVERED"
            filename = "snow_covered"

        case "SNOW FREE":
            cmap = "Greens"
            cbar_title = "Snow free [%]"
            vmin = 0
            vmax = 100
            title = "SNOW FREE"
            filename = "snow_free"

        case "L1C":
            title = "SENTINEL-2 L1C"
            filename = "l1c"

    # Title
    if plot_title:
        plt.title(title, fontsize="x-large", pad=10, loc="left", fontweight="bold")

        if sle is not None and len(sle) == 1:
            plt.title(f"SLE = {int(sle[0])} m", fontsize="large", pad=10, loc="right")

    if image_type == "L1C":
        ax.imshow(image)

    else:
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="3%", pad=0.2)
        cax.set_aspect(0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.set_title(cbar_title)

    if sle is not None:
        if len(sle) == 1:
            ax.contour(dem, levels=sle, colors="deeppink", zorder=50, linewidths=2)
            filename = f"snowline_{filename}"

        if len(sle) == 2:
            ax.contour(dem, levels=sle, colors=["turquoise", "deeppink"], zorder=50, linewidths=1.5)
            filename = f"snowline_comparison_{filename}"
            gray_line = mlines.Line2D([], [], color="turquoise", label="Median February SLE (1789 m a.s.l.)")
            pink_line = mlines.Line2D([], [], color="deeppink", label="February 2022 SLE (2134 m a.s.l.)")
            ax.legend(handles=[gray_line, pink_line], loc="lower center")

    ax.axis("off")
    fig.tight_layout()

    if save:
        plt.savefig(f"{fig_dir}{filename}", dpi=300)

    plt.show()


def plot_time_series(basin_names_list, basin_snow_metrics_dict, indicator, save=False):
    """
    Plots time series of the specified snow indicator for the given basins.

    :param basin_names_list: List of basin names.
    :param basin_snow_metrics_dict: Dictionary containing snow metrics for each basin.
    :param indicator: Indicator to plot.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Match indicator
    match indicator:
        case "SLE (m a.s.l.)":
            indicator_label = indicator
            vmin = 300
            vmax = 3700
            indicator_filename = "sle"
        case "Snow cover (%)":
            indicator_label = indicator
            vmin = -10
            vmax = 110
            indicator_filename = "snow_covered_area"
        case "Mean snow depth (cm)":
            indicator_label = indicator
            vmin = -20
            vmax = 170
            indicator_filename = "mean_snow_depth"
        case "Mean SWE (mm)":
            indicator_label = indicator
            vmin = -50
            vmax = 550
            indicator_filename = "mean_swe"
        case "Rep. index (%)":
            indicator_label = "Representativeness index (%)"
            vmin = 34
            vmax = 106
            indicator_filename = "ri"
        case "Err. index (%)":
            indicator_label = "Error index (%)"
            vmin = -5
            vmax = 55
            indicator_filename = "ei"

    # Figure settings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = get_colors(len(basin_names_list))

    # Plot
    for i, basin_name in enumerate(basin_names_list):
        snow_metrics = basin_snow_metrics_dict[basin_name]
        year_month = pd.to_datetime(snow_metrics["YYYY-MM"])
        snow_metrics_indicator = snow_metrics[indicator]

        ax.plot(year_month, snow_metrics_indicator, label=basin_name, color=colors[i])

    # Title
    if plot_title:
        plt.title(indicator_label, fontsize="x-large", pad=10, loc="left")

    # x-axis settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Month")

    # y-axis settings
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(indicator_label)

    # Legend
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.11, right=0.72, top=0.96, bottom=0.14)

    if save:
        plt.savefig(f"{fig_dir}time_series/{indicator_filename}_time_series", dpi=300)

    plt.show()


def plot_time_series_deviation(basin_names_list, deviations_anomalies_dict, indicator, save=False):
    """
    Plots time series of the specified snow indicator for the given basins.

    :param basin_names_list: List of basin names.
    :param deviations_anomalies_dict: Dictionary containing deviations and anomalies data for each basin and indicator.
    :param indicator: The indicator to plot the time series deviation for.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Match indicator
    match indicator:
        case "SLE (m a.s.l.)":
            indicator_label = "SLE deviation (m)"
            vmin = -1100
            vmax = 1100
            indicator_filename = "sle"

    # Figure settings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = get_colors(len(basin_names_list))

    # Plot
    for i, basin_name in enumerate(basin_names_list):
        deviations_anomalies = deviations_anomalies_dict[basin_name][indicator]
        year_month = pd.to_datetime(deviations_anomalies["YYYY-MM"])
        deviation = deviations_anomalies["deviation"]

        ax.plot(year_month, deviation, label=basin_name, color=colors[i])

    # Title
    if plot_title:
        plt.title(indicator_label, fontsize="x-large", pad=10, loc="left")

    # x-axis settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Month")

    # y-axis settings
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(indicator_label)

    # Legend
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.11, right=0.72, top=0.96, bottom=0.14)

    if save:
        plt.savefig(f"{fig_dir}time_series/{indicator_filename}_time_series_deviation", dpi=300)

    plt.show()


def plot_deviation_monthly(basin_names_list, deviations_anomalies_dict, indicator, month, mode, save=False):
    """
    Plots deviation time series of the specified snow indicator and month for the given basins.

    :param basin_names_list: List of basin names.
    :param deviations_anomalies_dict: Dictionary containing deviations and anomalies data for each basin and indicator.
    :param indicator: Indicator to be plotted.
    :param month: Month to be selected in the plot.
    :param mode: Mode of the plot, either "deviation" or "anomaly".
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Match indicator
    match indicator:
        case "SLE (m a.s.l.)":
            indicator_label = "SLE"
            indicator_filename = "sle"

    # Match mode
    match mode:
        case "deviation":
            mode_label = "deviation (m)"
            vmin = -1100
            vmax = 1100
            mode_filename = "deviation"
        case "anomaly":
            mode_label = "anomaly (%)"
            vmin = -110
            vmax = 110
            mode_filename = "anomaly"

    # Figure settings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = get_colors(len(basin_names_list))

    # Plot
    for i, basin_name in enumerate(basin_names_list):
        deviations_anomalies = deviations_anomalies_dict[basin_name][indicator]
        year_month = deviations_anomalies["YYYY-MM"]
        data = deviations_anomalies[mode]

        # Get months
        year_month_selected = pd.to_datetime(year_month[year_month.str[-2:] == month])
        data_selected = data[year_month.str[-2:] == month]

        ax.plot(year_month_selected, data_selected, label=basin_name, color=colors[i])

    # Title
    if plot_title:
        plt.title(f"{indicator_label} {mode_label}", fontsize="x-large", pad=10, loc="left")
        plt.title(pl.parse(f"2000-{month}-01").format("MMMM"), fontsize="large", pad=10, loc="right")

    # x-axis settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=int(month)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel(pl.parse(f"2000-{month}-01").format("MMMM"))

    # y-axis settings
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(f"{indicator_label} {mode_label}")

    # Legend
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left")

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.11, right=0.72, top=0.96, bottom=0.14)

    if save:
        plt.savefig(f"{fig_dir}time_series/{indicator_filename}_{mode_filename}_{month}", dpi=300)

    plt.show()


def plot_deviation_bars(basin_names_list, deviations_anomalies_dict, indicator, timerange, months, mode, save=False):
    """
    Plots deviation bars for specified basins, indicators, time range, months, and mode.

    :param basin_names_list: List of basin names.
    :param deviations_anomalies_dict: Dictionary containing deviations and anomalies data.
    :param indicator: Indicator label.
    :param timerange: Time range for data.
    :param months: List of months.
    :param mode: Mode of the plot ("deviation" or "anomaly").
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Match indicator
    match indicator:
        case "SLE (m a.s.l.)":
            indicator_label = "SLE"
            indicator_filename = "sle"

    # Match mode
    match mode:
        case "deviation":
            mode_label = "deviation (m)"
            vmin = -1100
            vmax = 1100
            mode_filename = "deviation"
        case "anomaly":
            mode_label = "anomaly (%)"
            vmin = -110
            vmax = 110
            mode_filename = "anomaly"

    # Extract data
    year = timerange[1][:-3]

    deviations = {}
    for month in months:
        deviations[month] = []

    labels = []

    for basin_name in basin_names_list:
        deviation_indicator = deviations_anomalies_dict[basin_name][indicator]

        idx_start = deviation_indicator[deviation_indicator["YYYY-MM"] == timerange[0]].index.values[0]
        idx_end = deviation_indicator[deviation_indicator["YYYY-MM"] == timerange[1]].index.values[0]
        deviation_indicator_valid = deviation_indicator.iloc[idx_start:idx_end + 1].reset_index(drop=True)

        for month in months:
            if int(month) >= 10:
                year_month = f"{int(year) - 1}-{month}"
            else:
                year_month = f"{year}-{month}"

            labels.append(pl.parse(f"{year_month}-01").format("MMM YYYY"))
            deviation = deviation_indicator_valid[deviation_indicator_valid["YYYY-MM"] == year_month][mode].values[0]
            deviations[month].append(deviation)

    # Figure settings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = get_colors(len(months))

    # Plot
    width = 0.8 / len(months)
    x = np.arange(len(basin_names_list))

    for i, month in enumerate(months):
        ax.bar(x - ((len(months) - 1) / 2) * width + i * width, deviations[month],
               color=colors[i], width=width, label=labels[i], zorder=50)

    # Title
    if plot_title:
        plt.title(f"{indicator_label} {mode_label}", fontsize="x-large", pad=10, loc="left")
        plt.title(f"Winter season {int(year) - 1}/{year[-2:]}", fontsize="large", pad=10, loc="right")

    # x-axis settings
    ax.set_xlim(-0.5, 8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([basin_name.split(" / ")[0] for basin_name in basin_names_list])

    # y-axis settings
    ax.set_ylim(vmin, vmax)
    ax.set_ylabel(f"{indicator_label} {mode_label}")

    # Legend
    ax.legend(ncol=len(months), loc="lower center", bbox_to_anchor=(0.5, 1))

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.89, bottom=0.09)

    if save:
        plt.savefig(f"{fig_dir}time_series/{indicator_filename}_{mode_filename}_{int(year)}", dpi=300)

    plt.show()


def plot_time_series_comparison(basin_name, basin_snow_metrics_dict, monthly_stats_dict, deviations_anomalies_dict,
                                timerange, save=False):
    """
    Plots SLE, median SLE and percentiles together with SLE deviations and SWE deviations.

    :param basin_name: The name of the basin.
    :param basin_snow_metrics_dict: A dictionary containing snow metrics data for each basin.
    :param monthly_stats_dict: A dictionary containing monthly statistics data for each basin.
    :param deviations_anomalies_dict: A dictionary containing deviations and anomalies data for each basin.
    :param timerange: The time range for which the time series comparison will be plotted.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    fig, ax = plt.subplots(3, 1, figsize=(8.2, 5.5))

    year = timerange[1][:-3]

    snow_metrics = basin_snow_metrics_dict[basin_name]
    monthly_stats = monthly_stats_dict[basin_name]
    deviations_anomalies = deviations_anomalies_dict[basin_name]
    idx_start = snow_metrics[snow_metrics["YYYY-MM"] == timerange[0]].index.values[0]
    idx_end = snow_metrics[snow_metrics["YYYY-MM"] == timerange[1]].index.values[0]

    snow_metrics_timerange = snow_metrics.iloc[idx_start:idx_end + 1]
    year_month = pd.to_datetime(snow_metrics_timerange["YYYY-MM"])

    monthly_stats_indicator = monthly_stats["SLE (m a.s.l.)"]
    monthly_stats_timerange = pd.concat([monthly_stats_indicator,
                                         monthly_stats_indicator,
                                         monthly_stats_indicator,
                                         monthly_stats_indicator,
                                         monthly_stats_indicator])[:snow_metrics_timerange.shape[0]]

    deviations_anomalies_indicator_1 = deviations_anomalies["SLE (m a.s.l.)"]
    deviations_anomalies_timerange_1 = deviations_anomalies_indicator_1.iloc[idx_start:idx_end + 1]

    deviations_anomalies_indicator_2 = deviations_anomalies["Mean SWE (mm)"]
    deviations_anomalies_timerange_2 = deviations_anomalies_indicator_2.iloc[idx_start:idx_end + 1]

    # Plot SLE ---------------------------------------------------------------------------------------------------------
    ax[0].plot(year_month, snow_metrics_timerange["SLE (m a.s.l.)"], color="deeppink", label="SLE", zorder=50)

    ax[0].plot(year_month, monthly_stats_timerange["MEDIAN"], color="gray", zorder=20, label="Median SLE 2017-2022")

    ax[0].fill_between(year_month, monthly_stats_timerange["P10"], monthly_stats_timerange["P90"],
                       color="lightgray", alpha=0.8, edgecolor=None, zorder=10, label="10/90 perc. 2017-2022")

    # x-axis settings
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[0].set_xticklabels([])

    # y-axis settings
    ax[0].set_ylabel("SLE (m a.s.l.)")
    ax[0].set_ylim(300, 4200)
    ax[0].set_yticks([500, 1500, 2500, 3500])

    # Legend
    ax[0].legend(loc="upper center", ncol=3)

    # General
    ax[0].grid()
    remove_spines(ax[0])

    # Plot SLE deviation ---------------------------------------------------------------------------------------------
    colors = np.where(deviations_anomalies_timerange_1["deviation"] >= 0, "mediumseagreen", "indianred")
    ax[1].bar(year_month, deviations_anomalies_timerange_1["deviation"], width=20, zorder=50, color=colors)

    # x-axis settings
    ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[1].set_xticklabels([])
    ax[1].set_xlim(ax[0].get_xlim())

    # y-axis settings
    ax[1].set_ylabel("SLE deviation (m)")
    ax[1].set_ylim(-960, 960)
    ax[1].set_yticks([-800, -400, 0, 400, 800])

    # General
    ax[1].grid()
    remove_spines(ax[1])

    # Plot SWE deviation ---------------------------------------------------------------------------------------------
    colors = np.where(deviations_anomalies_timerange_2["deviation"] >= 0, "mediumseagreen", "indianred")
    ax[2].bar(year_month, deviations_anomalies_timerange_2["deviation"], width=20, zorder=50, color=colors)

    # x-axis settings
    ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax[2].set_xlim(ax[0].get_xlim())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax[2].set_xlabel("Month")

    # y-axis settings
    ax[2].set_ylabel("SWE deviation (mm)")
    ax[2].set_ylim(-240, 240)
    ax[2].set_yticks([-200, -100, 0, 100, 200])

    # General
    ax[2].grid()
    remove_spines(ax[2])

    plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12)

    if save:
        plt.savefig(f"{fig_dir}time_series/comparison_time_series_{convert_to_snake_case(basin_name)}", dpi=300)

    plt.show()


def plot_grid(basin_names_list, deviations_anomalies_dict, indicators, month, save=False):
    """
    Plots two anomaly grids (SLE and SLE) for all basins in the specified month.

    :param basin_names_list: A list of basin names.
    :param deviations_anomalies_dict: A dictionary containing the deviations and anomalies data for each basin and indicator.
    :param indicators: A list of indicator names.
    :param month: The month for which the grid will be plotted.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Create grid
    grids = {}
    for indicator in indicators:
        grids[indicator] = np.zeros((8, 7))

    for indicator, grid in grids.items():
        for i, basin_name in enumerate(basin_names_list):
            deviations_anomalies = deviations_anomalies_dict[basin_name][indicator]
            year_month = deviations_anomalies["YYYY-MM"]
            deviations = deviations_anomalies["anomaly"]

            deviations_month = deviations[year_month.str[-2:] == month].values

            grids[indicator][i, :] = deviations_month

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))

    filename = []

    for i, indicator in enumerate(indicators):
        match indicator:
            case "SLE (m a.s.l.)":
                indicator_label = "SLE (inverted)"
                cmap = "PiYG_r"
                indicator_filename = "sle"
            case "Snow cover (%)":
                indicator_label = "Snow cover"
                cmap = "PiYG"
                indicator_filename = "snow_cover"
            case "Mean snow depth (cm)":
                indicator_label = "Snow depth"
                cmap = "PiYG"
                indicator_filename = "snow_depth"
            case "Mean SWE (mm)":
                indicator_label = "SWE"
                cmap = "PiYG"
                indicator_filename = "swe"

        filename.append(indicator_filename)

        img = ax[i].imshow(grids[indicator], cmap=cmap)

        ax[i].set_title(indicator_label, fontsize="large", pad=10)

        # x-axis settings
        ax[i].set_xticks(np.arange(grids[indicator].shape[1]))
        ax[i].set_xticklabels(["2017", "2018", "2019", "2020", "2021", "2022", "2023"], rotation=30)
        ax[i].set_xlabel(pl.parse(f"2000-{month}-01").format("MMMM"))

        # y-axis settings
        if i == 0:
            ax[i].set_yticks(np.arange(len(basin_names_list)))
            ax[i].set_yticklabels(basin_names_list)
        else:
            ax[i].set_yticklabels([])

        # General
        remove_spines(ax[i], ticks_off=False)
        ax[i].tick_params(axis="y", length=0)

    plt.subplots_adjust(left=0.1, right=1.025, top=0.9, bottom=0.2, wspace=-0.2)

    # Colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", pad=0.04, shrink=1)
    cbar.ax.set_ylabel("Anomaly (%)", labelpad=5)

    if save:
        plt.savefig(f"{fig_dir}time_series/{filename[0]}_{filename[1]}_anomaly_grid_{month}", dpi=300)

    plt.show()


def plot_scattering_time_series(basin_name, basin_snow_metrics_dict, mode, save=False):
    """
    Plots time series of the distribution of the erroneous pixels from the SLE estimation.

    :param basin_name: Name of the basin.
    :param basin_snow_metrics_dict: Dictionary of basin snow metrics data.
    :param mode: Mode indicating whether to plot "below" or "above" the snow line.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Figure settings
    fig, ax = plt.subplots(figsize=(8.2, 4))
    colors = get_colors(2)

    snow_metrics = basin_snow_metrics_dict[basin_name]
    year_month = pd.to_datetime(snow_metrics["YYYY-MM"])

    match mode:
        case "below":
            data = snow_metrics["Snow covered below - south (%)"]
            y_label = "Snow covered pixels below SLE (%)"
        case "above":
            data = snow_metrics["Snow free above - south (%)"]
            y_label = "Snow free pixels above SLE (%)"

    # Plot
    ax.fill_between(year_month, data, 100, color=colors[1], alpha=0.5, zorder=50, label="North faces")
    ax.fill_between(year_month, data, color=colors[0], alpha=0.5, zorder=50, label="South faces")

    # x-axis settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Month")
    ax.set_xlim(year_month.iloc[0], year_month.iloc[-1])

    # y-axis settings
    ax.set_ylim(0, 100)
    ax.set_ylabel(y_label)

    # Legend
    ax.legend(loc="lower right").set_zorder(100)

    # General
    ax.grid()
    remove_spines(ax, ticks_off=False)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.18)

    if save:
        plt.savefig(f"{fig_dir}time_series/scattering_{mode}_time_series", dpi=300)

    plt.show()


def plot_test_areas(test_areas_shapefile, image_path, save=False):
    """
    Plots the test areas on top of a satellite image.

    :param test_areas_shapefile: Path to the shapefile containing the test areas.
    :param image_path: Path to the image file.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8.2, 5))

    with rasterio.open(image_path, "r") as img:
        show(img.read(), transform=img.transform, ax=ax, alpha=0.8)

    # Load test areas
    test_areas = gpd.read_file(test_areas_shapefile).to_crs(3035)

    # Plot test areas
    test_areas.plot(ax=ax, edgecolor="black", facecolor=(0.196, 0.804, 0.196, 0.4), linewidth=1.25)

    # Get annotation coordinates
    test_areas["coords"] = test_areas["geometry"].apply(lambda x: x.representative_point().coords[:])
    test_areas["coords"] = [coords[0] for coords in test_areas["coords"]]

    # Annotate
    for idx, row in test_areas.iterrows():
        plt.annotate(row["id"], xy=row["coords"], ha="center", va="center", color="black")

    ax.axis("off")
    fig.tight_layout()

    if save:
        plt.savefig(f"{fig_dir}gfsc_validation/test_areas", dpi=300)

    plt.show()


def plot_l2a_custom_snow_cover(l2a, custom_snow_cover, label, save=False):
    """
    Plots Sentinel-2 L2A imagery and custom binary snow cover.

    :param l2a: The Sentinel-2 Level-2A image.
    :param custom_snow_cover: The custom snow cover mask.
    :param label: The label for the plot.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 4))

    # Left plot
    ax[0].imshow(l2a)
    ax[0].set_title("Sentinel-2 L2A", fontsize="large", pad=10)

    # Right plot
    ax[1].imshow(l2a)
    ax[1].imshow(np.where(custom_snow_cover == 0, np.nan, custom_snow_cover), cmap="plasma", vmin=0, vmax=100)
    ax[1].set_title("Custom snow cover", fontsize="large", pad=10)

    remove_spines(ax[0], tick_labels_off=True, border_off=False)
    remove_spines(ax[1], tick_labels_off=True, border_off=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.05)

    if save:
        plt.savefig(f"{fig_dir}gfsc_validation/s2l2a_custom_sc_test_area_{label}", dpi=300)

    plt.show()


def plot_gfsc_custom_fsc(gfsc, custom_fsc, label, save=False):
    """
    Plots GFSC and custom FSC.

    :param gfsc: 2D array representing the HR-S&I GFSC values.
    :param custom_fsc: 2D array representing the custom FSC values.
    :param label: string representing the label for the plot.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Left plot
    gfsc = ax[0].imshow(gfsc, cmap="Blues", vmin=0, vmax=100)
    ax[0].set_title("HR-S&I GFSC", fontsize="large", pad=10)

    # Right plot
    ax[1].imshow(custom_fsc, cmap="Blues", vmin=0, vmax=100)
    ax[1].set_title("Custom FSC", fontsize="large", pad=10)

    remove_spines(ax[0], tick_labels_off=True, border_off=False)
    remove_spines(ax[1], tick_labels_off=True, border_off=False)
    plt.subplots_adjust(left=0.045, right=1.025, top=0.94, bottom=0.01, wspace=0.1)

    # Colorbar
    cbar = fig.colorbar(gfsc, ax=ax, orientation="vertical", pad=0.04, shrink=0.915)
    cbar.ax.set_ylabel("Fractional snow cover (%)", labelpad=5)

    if save:
        plt.savefig(f"{fig_dir}gfsc_validation/gfsc_custom_fsc_test_area_{label}", dpi=300)

    plt.show()


def plot_l2a_diff(l2a, gfsc, custom_fsc, label, save=False):
    """
    Plots Sentinel-2 L2A imagery and difference between GFSC and custom FSC.

    :param l2a: The Sentinel-2 Level 2A image.
    :param gfsc: The HR-S&I GFSC image.
    :param custom_fsc: The custom FSC image.
    :param label: The label for the saved figure.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: A dictionary containing the metrics: MAE (mean absolute error),
             RMSE (root mean squared error), PSNR (peak signal to noise ratio),
             SSIM (structural similarity index).
    """

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    # Stats
    gfsc_filtered = gfsc[~np.isnan(gfsc)].flatten()
    custom_fsc_filtered = custom_fsc[~np.isnan(gfsc)].flatten()

    mae = mean_absolute_error(gfsc_filtered, custom_fsc_filtered)
    rmse = mean_squared_error(gfsc_filtered, custom_fsc_filtered, squared=False)
    psnr = peak_signal_noise_ratio(gfsc_filtered, custom_fsc_filtered, data_range=100)
    ssim = structural_similarity(gfsc_filtered, custom_fsc_filtered, data_range=100)

    # Left plot
    ax[0].imshow(l2a)
    ax[0].set_title("Sentinel-2 L2A", fontsize="large", pad=10)

    # Right plot
    diff = ax[1].imshow(gfsc - custom_fsc, cmap="PiYG", vmin=-100, vmax=100)
    ax[1].set_title("HR-S&I GFSC - Custom FSC", fontsize="large", pad=10)

    remove_spines(ax[0], tick_labels_off=True, border_off=False)
    remove_spines(ax[1], tick_labels_off=True, border_off=False)
    plt.subplots_adjust(left=0.045, right=1.025, top=0.94, bottom=0.01, wspace=0.1)

    # Colorbar
    cbar = fig.colorbar(diff, ax=ax, orientation="vertical", pad=0.04, shrink=0.915)
    cbar.ax.set_ylabel("GFSC - Custom FSC difference (%)", labelpad=5)

    if save:
        plt.savefig(f"{fig_dir}gfsc_validation/s2l2a_diff_test_area_{label}", dpi=300)

    plt.show()

    return {"MAE": mae, "RMSE": rmse, "PSNR": psnr, "SSIM": ssim}


def plot_diff_grid(gfsc_list, custom_fsc_list, save=False):
    """
    Plots a grid of differences between GFSC and custom FSC data.

    :param gfsc_list: A list of GFSC (Gap-filled Fractional Snow Cover) data.
    :param custom_fsc_list: A list of custom FSC (Fractional Snow Cover) data.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    fig, ax = plt.subplots(2, 4, figsize=(9, 4.5))

    for i, diff in enumerate(gfsc_list):
        row = i // 4  # Calculates the row index
        col = i % 4  # Calculates the column index
        diff = ax[row, col].imshow(gfsc_list[i] - custom_fsc_list[i], cmap="PiYG", vmin=-100, vmax=100)
        ax[row, col].set_title(f"Test area {i + 1}")
        remove_spines(ax[row, col], tick_labels_off=True, border_off=False)

    plt.subplots_adjust(left=0.03, right=1.025, top=0.94, bottom=0.015, wspace=0.1)

    # Colorbar
    cbar = fig.colorbar(diff, ax=ax, orientation="vertical", pad=0.04, shrink=0.95)
    cbar.ax.set_ylabel("GFSC - Custom FSC difference (%)", labelpad=5)

    if save:
        plt.savefig(f"{fig_dir}gfsc_validation/diff_grid", dpi=300)

    plt.show()


def plot_snow_depth_time_series(stations_df, ground_snow_depth_df, dta_snow_depth_df, station_name, save=False):
    """
    Plots the time series of ground snow depth and DTA snow depth for a specified station.

    :param stations_df: DataFrame containing information about the stations (name, elevation, sample size, RMSE, and correlation)
    :param ground_snow_depth_df: DataFrame containing ground snow depth data.
    :param dta_snow_depth_df: DataFrame containing DTA snow depth data.
    :param station_name: The name of the station to plot the time series for.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Figure settings
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = get_colors(2)

    for i in range(len(stations_df["Name"])):
        name = stations_df.loc[i, "Name"]
        elevation = stations_df.loc[i, "Elevation"]
        ss = stations_df.loc[i, "Sample size"]
        rmse = stations_df.loc[i, "RMSE"]
        r = stations_df.loc[i, "Correlation"]

        if station_name == name:
            # Get max values
            ymax = np.round(
                np.ceil(np.maximum(np.max(ground_snow_depth_df.iloc[i]), np.max(dta_snow_depth_df.iloc[i]))), -1)

            # Convert to datetime
            dates = pd.to_datetime(ground_snow_depth_df.columns)

            # Plot
            ax.plot(dates, ground_snow_depth_df.iloc[i], color=colors[0], label="Ground snow depth")
            ax.plot(dates, dta_snow_depth_df.iloc[i], color=colors[1], label="DTA snow depth")

            # Add stats
            ax.annotate(f"RMSE = {round(rmse, 2)} cm\nr = {round(r, 2)}\nSample size = {int(ss)}",
                        (0.03, 0.965), xycoords="axes fraction", ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white"), fontsize="medium")

            # Plot station name and elevation
            plt.title(f"{name}", fontsize="large", pad=10, loc="left")
            plt.title(f"{elevation} m a.s.l.", fontsize="large", pad=10, loc="right")

            # Filename
            filename = f"station_{i + 1}"

    # x-axis settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date")

    # y-axis settings
    ax.set_ylim(-4, ymax + 4)
    ax.set_ylabel("Snow depth (cm)")

    # Legend
    ax.legend(loc="upper right")

    # General
    ax.grid()
    remove_spines(ax)
    plt.subplots_adjust(left=0.1, right=0.96, top=0.88, bottom=0.15)

    if save:
        plt.savefig(f"{fig_dir}snow_depth_validation/sd_time_series_{filename}", dpi=300)

    plt.show()


def plot_snow_depth_scatter(stations, ground_snow_depth_list, dta_snow_depth_list, station=None, mode=None,
                            drop_stations=False, save=False):
    """
    Plots scatter plots of ground snow depth against DTA snow depth for one or all stations and calculates statistics.

    :param stations: DataFrame containing information about the stations.
    :param ground_snow_depth_list: List of NumPy arrays containing ground snow depth values for each station.
    :param dta_snow_depth_list: List of NumPy arrays containing DTA snow depth values for each station.
    :param station: Optional string specifying a single station to plot.
    :param mode: Optional string specifying the mode for coloring the scatter plot (options: "RMSE", "Correlation").
    :param drop_stations: Boolean value indicating whether to drop stations with negative correlation or high RMSE.
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    # Count total stations and stations excluded
    stations_total = 0
    stations_excluded = 0

    # Store not excluded station values
    ground_snow_depth_list_acc = []
    dta_snow_depth_list_acc = []

    fig, ax = plt.subplots(figsize=(8.2, 5))

    for i in range(len(stations["Name"])):
        name = stations.loc[i, "Name"]
        elevation = stations.loc[i, "Elevation"]
        ss = stations.loc[i, "Sample size"]
        rmse = stations.loc[i, "RMSE"]
        r = stations.loc[i, "Correlation"]
        stations_total += 1

        # Plot single station
        if station is not None:
            if station == name:
                # Get max values
                vmax = np.round(np.ceil(np.maximum(np.max(ground_snow_depth_list[i]), np.max(dta_snow_depth_list[i]))),
                                -1) + 4

                # Scatter plot
                ax.scatter(ground_snow_depth_list[i], dta_snow_depth_list[i], zorder=20, label=name,
                           edgecolors="royalblue", facecolors="none")

                # Calculate and plot the best-fit line (y = mx + b)
                m, b = np.polyfit(ground_snow_depth_list[i], dta_snow_depth_list[i], 1)
                best_fit_line = np.poly1d([m, b])

                x = np.linspace(-4, vmax)
                ax.plot(x, x, color="gray", linestyle="dashed", zorder=40)
                ax.plot(x, best_fit_line(x), color="orangered", zorder=50)

                ax.annotate(
                    f"RMSE = {round(rmse, 2)} cm\nr = {round(r, 2)}\nBest fit (y=mx+b):\nm = {round(m, 2)}, b = {round(b, 2)}\nSample size = {int(ss)}",
                    (0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white"), fontsize="medium", zorder=100)

                plt.title(f"{name}", fontsize="large", pad=10, loc="left")
                plt.title(f"{elevation} m a.s.l.", fontsize="large", pad=10, loc="right")

                filename = f"station_{i + 1}"

        # Plot all
        else:
            vmax = 224

            if mode is not None:
                match mode:
                    case "RMSE":
                        cmap = plt.cm.get_cmap("cool")
                        norm = mcolors.Normalize(0, 100)
                        norm_data = norm(rmse)
                        cbar_title = "Station RMSE (cm)"
                        mode_filename = "rmse"

                    case "Correlation":
                        cmap = plt.cm.get_cmap("RdYlGn")
                        norm = mcolors.Normalize(-1, 1)
                        norm_data = norm(r)
                        cbar_title = "Station r"
                        mode_filename = "r"

                color = cmap(norm_data)

            else:
                color = "royalblue"

            if drop_stations and (r < 0 or rmse > 50):
                # print(f"{name} excluded!")
                stations_excluded += 1

                # Filename
                filename = f"{mode_filename}_excluded"

            else:
                # Add valid data to list
                ground_snow_depth_list_acc.append(ground_snow_depth_list[i])
                dta_snow_depth_list_acc.append(dta_snow_depth_list[i])

                # Scatter plot
                ax.scatter(ground_snow_depth_list[i], dta_snow_depth_list[i],
                           zorder=20, edgecolors=color, facecolors="none", alpha=0.6,
                           label=f"{name} ({elevation} m)")

                # Filename
                filename = mode_filename

    # Calculate overall stats
    if station is None:
        if drop_stations:
            ground_snow_depth_list_flat = np.array([item for sublist in ground_snow_depth_list_acc for item in sublist])
            dta_snow_depth_list_flat = np.array([item for sublist in dta_snow_depth_list_acc for item in sublist])

        else:
            ground_snow_depth_list_flat = np.array([item for sublist in ground_snow_depth_list for item in sublist])
            dta_snow_depth_list_flat = np.array([item for sublist in dta_snow_depth_list for item in sublist])

        # Calculate and plot the best-fit line (y = mx + b)
        m, b = np.polyfit(ground_snow_depth_list_flat, dta_snow_depth_list_flat, 1)
        best_fit_line = np.poly1d([m, b])

        x = np.linspace(-4, vmax)
        ax.plot(x, x, color="gray", linestyle="dashed", zorder=40)
        ax.plot(x, best_fit_line(x), color="orangered", zorder=50)

        # Sample size
        ss = ground_snow_depth_list_flat.size

        # RMSE
        differences = dta_snow_depth_list_flat - ground_snow_depth_list_flat
        rmse = np.sqrt(np.mean(differences ** 2))

        # Pearson
        correlation_coefficient = np.corrcoef(dta_snow_depth_list_flat, ground_snow_depth_list_flat)
        r = correlation_coefficient[0, 1]

        ax.annotate(
            f"RMSE = {round(rmse, 2)} cm\nr = {round(r, 2)}\nBest fit (y=mx+b): m = {round(m, 2)}, b = {round(b, 2)}\nSample size = {int(ss)}",
            (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white"), fontsize="medium", zorder=100)

        plt.title(f"Overall agreement", fontsize="large", pad=10, loc="left")
        plt.title(f"{stations_total - stations_excluded}/{stations_total} stations", fontsize="large", pad=10,
                  loc="right")

    if mode is not None:
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cax.set_aspect(30)
        cbar = fig.colorbar(scalar_map, cax=cax, orientation="vertical")
        cbar.ax.set_ylabel(cbar_title, labelpad=5)

    # x-axis settings
    ax.set_xlim(-4, vmax)
    ax.set_xlabel("Ground snow depth (cm)", labelpad=5)

    # y-axis settings
    ax.set_ylim(-4, vmax)
    ax.set_ylabel("DTA snow depth (cm)", labelpad=5)

    # General
    ax.grid()
    ax.set_aspect("equal")
    remove_spines(ax)
    plt.subplots_adjust(left=0, right=1, top=0.89, bottom=0.12)

    if save:
        plt.savefig(f"{fig_dir}/snow_depth_validation/sd_scatter_plot_{filename}", dpi=300)

    plt.show()


def plot_weather_stations(l1c, stations_df, stations_color=None, save=False):
    """
    Plots weather stations on top of a Sentinel-2 L1C image.

    :param l1c: 2D array containing the Sentinel-2 L1C data.
    :param stations_df: DataFrame containing the coordinates of weather stations.
    :param stations_color: Optional string indicating the type of colorization to use for the weather stations.
                           Can be "RMSE" or "Correlation".
    :param save: A boolean indicating whether to save the generated plot as an image file. Default is False.
    :return: None
    """

    fig, ax = plt.subplots(figsize=(8.2, 5))

    ax.imshow(l1c, alpha=0.7)

    if stations_color is None:
        ax.scatter(stations_df["col"], stations_df["row"], zorder=100, edgecolors="black", s=75,
                   facecolors="limegreen")
        filename = "weather_stations"

    else:
        match stations_color:
            case "RMSE":
                cmap = "cool"
                cbar_title = "RMSE"
                vmin = 0
                vmax = 100
                ticks = [0, 20, 40, 60, 80, 100]
                filename = "weather_stations_rmse"
            case "Correlation":
                cmap = "RdYlGn"
                cbar_title = "r"
                vmin = -1
                vmax = 1
                ticks = [-1, 0, 1]
                filename = "weather_stations_r"

        scat = ax.scatter(stations_df["col"], stations_df["row"], zorder=100, edgecolors="black", s=75,
                          c=stations_df[stations_color], cmap=cmap, vmin=vmin, vmax=vmax)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="2%", pad=-0.4)
        cax.set_aspect(0.04)
        cbar = fig.colorbar(scat, cax=cax, orientation="horizontal", ticks=ticks)
        cbar.ax.set_title(cbar_title)

    for i, station in stations_df.iterrows():
        ax.annotate(i + 1, (station["col"] - 15, station["row"] - 25), va="center", ha="center",
                    color="black", zorder=110)

    ax.axis("off")
    fig.tight_layout()

    if save:
        plt.savefig(f"{fig_dir}snow_depth_validation/{filename}", dpi=300)

    plt.show()
