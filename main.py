from analysis import *

start_time = time.time()
save_plots = False

# Setup study area
basin_names_list = ["Tanaro", "Maira / Dora Riparia", "Dora Baltea", "Sesia",
                    "Ticino / Lago Maggiore", "Adda / Lago di Como", "Oglio", "Mincio"]

# Get geometries
hydrobasins_path = "data/shapefiles/hydrobasins_europe.shp"
alpine_perimeter_path = "data/shapefiles/alpine_convention_perimeter.shp"
geometries = get_basins(hydrobasins_path, alpine_perimeter_path, basin_names_list, 3035)

# Elevation histograms
dem_analysis(geometries, save_plots=save_plots)

# SLE estimation (Dora Baltea, 24-01-2022)
metrics = main_analysis("Dora Baltea", "2022-01-24", "2022-01-24", geometries, plot=True, save_plots=save_plots)

# Snow metrics estimation (October 2016 - June 2023) and time series analysis
start_date = "2016-10-01"
end_date = "2023-06-30"

for basin_name in basin_names_list:
    main_analysis(basin_name, start_date, end_date, geometries)

time_series_analysis(basin_names_list, start_date, end_date, geometries, save_plots=save_plots)

# GFSC accuracy assessment (April 14, 2022)
gfsc_accuracy_assessment("data/shapefiles/test_areas.shp", "2022-04-14", save_plots=save_plots)

# DTA snow depth validation (November 2021 - April 2022)
snow_depth_analysis("Dora Baltea", geometries, "2021-11-01", "2022-04-30")
snow_depth_validation("Dora Baltea", geometries, save_plots=save_plots)

print(f"Total runtime: {round((time.time() - start_time) / 60, 3)} minutes")
