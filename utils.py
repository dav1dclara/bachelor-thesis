import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pendulum as pl
import richdem as rd
import rasterio
from rasterio import warp
from rasterio.plot import show
import skimage.measure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sentinelhub import (
    Geometry,
    CRS,
    SHConfig,
    SentinelHubRequest,
    MimeType,
    DataCollection
)


def convert_to_snake_case(name):
    """
    Converts a string to snake case.

    :param name: The string to be converted.
    :return: The string in snake case.
    """
    words = name.replace("/", '').split()
    words_lower = [word.lower() for word in words]
    snake_case_name = "_".join(words_lower)

    return snake_case_name


def get_year_month_list(start_date, end_date):
    """
    Takes a start_date and end_date as input and returns a list of year-month strings, formatted as "YYYY-MM".

    :param start_date: Start date in the format "YYYY-MM-DD".
    :param end_date: End date in the format "YYYY-MM-DD".
    :return: List of year-month strings in the format "YYYY-MM" between start_date and end_date (inclusive).
    """
    start_date = pl.parse(start_date)
    end_date = pl.parse(end_date)

    months = []

    current_date = start_date
    while current_date <= end_date:
        months.append(current_date.format("YYYY-MM"))
        current_date = current_date.add(months=1)

    return months


def get_date_list(start_date, end_date):
    """
    Generates a list of dates between two given dates.

    :param start_date: The start date in "YYYY-MM-DD" format.
    :param end_date: The end date in "YYYY-MM-DD" format.
    :return: A list of dates in "YYYY-MM-DD" format.
    """
    start_date = pl.parse(start_date)
    end_date = pl.parse(end_date)

    dates = []

    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.format("YYYY-MM-DD"))
        current_date = current_date.add(days=1)

    return dates


def get_first_and_last_day_of_month(year_month):
    """
    Get the first and last day of a given month.

    :param year_month: The year and month in the format "YYYY-MM".
    :return: A tuple of the first day and last day of the given month in the format "YYYY-MM-DD".
    """
    date = pl.parse(year_month)
    first_day = date.start_of("month").format("YYYY-MM-DD")
    last_day = date.end_of("month").format("YYYY-MM-DD")

    return first_day, last_day


def get_name_of_month(year_month):
    """
    Takes a year-month string and returns the name of the month.

    :param year_month: A string representing the year and month in the format "YYYY-MM".
    :return: The name of the month corresponding to the given year and month.
    """
    date = pl.parse(year_month)

    return date.format("MMMM")


def downsample(img, block_size):
    """
    Takes an image and a block size and downsamples the image by averaging the pixel values within each block.

    :param img: The input image to be downsampled.
    :param block_size: The size of the blocks used for downsampling.
    :return: The downsampled image.
    """
    return skimage.measure.block_reduce(img, block_size, np.mean)


def get_aspect(dem):
    """
    Calculate the aspect of a digital elevation model (DEM).

    :param dem: A 2D array representing the DEM.
    :return: A 2D array representing the aspect of the DEM.
    """
    dem_rd = rd.rdarray(dem, no_data=np.nan)
    aspect_rd = rd.TerrainAttribute(dem_rd, attrib="aspect")

    return aspect_rd.view(np.ndarray)


def get_colors(n):
    """
    Returns a list of colors using seaborn's color palette.

    :param n: The number of colors to generate.
    :return: A list of colors.
    """
    return sns.color_palette(n_colors=n)


def truncate_colormap(cmap, minval=0, maxval=1, n=100):
    """
    Truncates the given colormap by specifying the lower and upper bounds and the number of colors.

    :param cmap: The colormap to truncate.
    :param minval: The lower bound of the colormap to truncate. Default is 0.
    :param maxval: The upper bound of the colormap to truncate. Default is 1.
    :param n: The number of colors in the truncated colormap. Default is 100.
    :return: The truncated colormap.
    """
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def remove_spines(ax, border_off=True, ticks_off=True, tick_labels_off=False):
    """
    Remove spines from an Axes object.

    :param ax: The Axes object.
    :param border_off: Whether to remove the border of the Axes. Default is True.
    :param ticks_off: Whether to remove ticks from both x and y axes. Default is True.
    :param tick_labels_off: Whether to remove tick labels from both x and y axes. Default is False.
    :return: None
    """
    if border_off:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if ticks_off:
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)

    if tick_labels_off:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
