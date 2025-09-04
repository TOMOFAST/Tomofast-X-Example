"""
A script for visualisation of Tomofast-x final model (using Python tools).

Author: Vitaliy Ogarko
"""

import os
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import cm

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.colorbar as cbar
from scipy.interpolate import griddata


import pandas as pd
import matplotlib.pyplot as pl
import numpy as np


# ==================================================================================================
# Visualisation of a 2D model slice.
def draw_model(grid, model, title, palette, to_file=None):
    """
    Draw the model slice.
    Note: the input grid should be a 2D grid corresponding to the model slice.
    """
    nelements = model.shape[0]

    pl.figure(figsize=(12.8, 9.6))

    # Makes the same scale for x and y axis.
    pl.axis("scaled")

    x_min = np.min(grid[:, 0:2])
    x_max = np.max(grid[:, 0:2])
    y_min = np.min(grid[:, 2:4])
    y_max = np.max(grid[:, 2:4])

    print("Model dimensions:", x_min, x_max, y_min, y_max)

    # Define figure dimensions.
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

    currentAxis = pl.gca()
    currentAxis.set_title(title)

    # Gradient palette.
    cmap = pl.get_cmap(palette)

    patches = []
    color_list = []

    # Use 5th and 95th percentiles for clipping
    val_min = np.percentile(model, 5)
    val_max = np.percentile(model, 95)

    for i in range(nelements):
        x1 = grid[i, 0]
        x2 = grid[i, 1]
        y1 = grid[i, 2]
        y2 = grid[i, 3]

        dx = x2 - x1
        dy = y2 - y1

        # Define the rectangle color.
        val = model[i]

        # Clip the value to the percentile range
        val_clipped = np.clip(val, val_min, val_max)

        if val_max != val_min:
            val_norm = (val_clipped - val_min) / (val_max - val_min)
        else:
            val_norm = 0.0
        color = cmap(val_norm)

        # Adding rectangle.
        patches.append(Rectangle((x1, y1), dx, dy))
        color_list.append(color)

    # Define patches collection with colormap.
    patches_cmap = ListedColormap(color_list)
    patches_collection = PatchCollection(patches, cmap=patches_cmap)
    patches_collection.set_array(np.arange(len(patches)))

    # Add rectangle collection to the figure.
    currentAxis.add_collection(patches_collection)

    # Show the colorbar.
    cax, _ = cbar.make_axes(currentAxis)
    # Set the correct colorbar scale using the percentile limits.

    norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)
    cb2 = cbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    if to_file is None:
        pl.show()
        pl.close(pl.gcf())
    else:
        pl.savefig(to_file)


# ==================================================================================================
# Visualisation of forward 1D data profile (along the model slice).
def draw_data(data_obs, data_calc, profile_coord, to_file=None):
    """
    Draw the data.

    profile_coord = 0, 1, 2, for x, y, z profiles.
    """
    # Increasing the figure size.
    # pl.figure(figsize=(12.8, 9.6))

    pl.plot(data_obs[:, profile_coord], data_obs[:, 3], "--bs", label="Observed data")
    pl.plot(
        data_calc[:, profile_coord], data_calc[:, 3], "--ro", label="Calculated data"
    )

    pl.legend(loc="upper left")

    if to_file is None:
        pl.show()
        pl.close(pl.gcf())
    else:
        pl.savefig(to_file)


# ==================================================================================================
# Visualisation of a 3D model.
def plot_3D_model(
    model, threshold, dzyx, filename="density", top_view=False, title="", to_file=None
):
    model = model.T  # transpose to match plotting orientation
    L, W, H = model.shape

    # Threshold mask
    filled = abs(model) >= threshold

    # Color and edgecolor arrays
    facecolors = np.empty(model.shape, dtype=object)
    edgecolors = np.empty(model.shape, dtype=object)

    # Color map setup
    norm = colors.Normalize(vmin=-1.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    # Assign colors to voxels above threshold
    for i, j, k in zip(*np.where(filled)):
        rgba = mapper.to_rgba(model[i, j, k])
        facecolors[i, j, k] = colors.rgb2hex(rgba)
        edgecolors[i, j, k] = "#000000"  # black edge (or rgba if desired)

    # Add dummy invisible voxels at 8 corners to enforce full bounding box
    corners = [
        (0, 0, 0),
        (L - 1, 0, 0),
        (0, W - 1, 0),
        (0, 0, H - 1),
        (L - 1, W - 1, 0),
        (L - 1, 0, H - 1),
        (0, W - 1, H - 1),
        (L - 1, W - 1, H - 1),
    ]
    for i, j, k in corners:
        filled[i, j, k] = True
        facecolors[i, j, k] = "#00000000"  # fully transparent face
        edgecolors[i, j, k] = "#00000000"  # fully transparent edge

    # Call the plotter
    pl_model_3D(
        filled,
        facecolors,
        dzyx,
        filename,
        top_view,
        edgecolors=edgecolors,
        title=title,
        to_file=to_file,
    )


# ==================================================================================================
# Visualisation of a 3D model (called by plot_3D_model).
def pl_model_3D(
    filled,
    facecolors,
    dzyx,
    filename="density",
    top_view=False,
    edgecolors=None,
    title="",
    to_file=None,
):
    fig = pl.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")

    # View settings
    ax.view_init(45, -45)
    if top_view:
        ax.set_proj_type("ortho")
        ax.view_init(90, -90)

    # Voxel grid coordinates
    x, y, z = np.indices(np.array(filled.shape) + 1)
    x = x * dzyx[2]
    y = y * dzyx[1]
    z = z * dzyx[0]

    # Plot voxels
    ax.voxels(
        x, y, z, filled, facecolors=facecolors, edgecolors=edgecolors, shade=False
    )

    # Axis formatting
    pl.axis("scaled")
    ax.invert_zaxis()
    ax.set_xlabel("X", labelpad=2)
    ax.set_ylabel("Y", labelpad=2)
    ax.set_zlabel("Z", labelpad=2)

    pl.title(title)
    if to_file is None:
        pl.show()
    else:
        pl.savefig(to_file)


# ==================================================================================================
# Visualisation of forward 2D data.
def plot_field(field, title, to_file=None):
    pl.figure(figsize=(6, 6), dpi=150)
    pl.imshow(field, cmap="jet", origin="lower")
    pl.colorbar()
    pl.title(title)
    if to_file is None:
        pl.show()
    else:
        pl.savefig(to_file)


# =====================================================================================================
def main(
    filename_model_grid,
    filename_model_final,
    filename_data_observed,
    filename_data_calculated,
    slice_index=1,
    slice_dim=0,
    palette="viridis",
    draw_true_model=True,
    to_folder=None,
):
    print("Started tomofast_vis.")

    # ----------------------------------------------------------------------------------
    # Setting the file paths and constants.
    # ----------------------------------------------------------------------------------

    # Path to input model grid (modelGrid.grav.file parameter in the Parfile).
    # filename_model_grid = '../Tomofast-x/data/gravmag/mansf_slice/true_model_grav_3litho.txt'

    # Path to the output model after inversion.
    # filename_model_final = '../Tomofast-x/output/mansf_slice/model/grav_final_model_full.txt'

    # Path to observed data (forward.data.grav.dataValuesFile parameter in the Parfile).
    # filename_data_observed = '../Tomofast-x/output/mansf_slice/data/grav_calc_read_data.txt'

    # Path to calculated data after inversion.
    # filename_data_calculated = '../Tomofast-x/output/mansf_slice/data/grav_calc_final_data.txt'

    # ----------------------------------------------------------------------------------
    # Reading data.
    # ----------------------------------------------------------------------------------
    if slice_dim == 0:
        plot_space_delimited_data(filename_data_observed, to_folder)

    # Reading the model grid.
    model_grid = np.loadtxt(
        filename_model_grid,
        dtype=float,
        usecols=(0, 1, 2, 3, 4, 5, 6),
        skiprows=1,
        delimiter=" ",
    )
    model_indexes = np.loadtxt(
        filename_model_grid, dtype=int, usecols=(7, 8, 9), skiprows=1
    )

    # Revert Z-axis.
    model_grid[:, 4] = -model_grid[:, 4]
    model_grid[:, 5] = -model_grid[:, 5]

    # Reading the final model.
    model_final = np.loadtxt(filename_model_final, dtype=float, skiprows=1)

    # Reading data.
    data_observed = np.loadtxt(
        filename_data_observed, dtype=float, usecols=(0, 1, 2, 3), skiprows=1
    )
    data_calculated = np.loadtxt(
        filename_data_calculated, dtype=float, usecols=(0, 1, 2, 3), skiprows=1
    )

    print("Ndata =", data_observed.shape[0])

    # ----------------------------------------------------------------------------------
    # Extract the model slices.
    # ----------------------------------------------------------------------------------

    # Extract the 2D profile.
    slice_filter = model_indexes[:, slice_dim] == slice_index

    model_grid_slice = model_grid[slice_filter]

    # When available the true model is stored in the grid file (7th column).
    true_model_slice = model_grid_slice[:, 6]

    # Grid slice dimensions.
    grid_slice_x_min = np.min(model_grid_slice[:, 0:2])
    grid_slice_x_max = np.max(model_grid_slice[:, 0:2])
    grid_slice_y_min = np.min(model_grid_slice[:, 2:4])
    grid_slice_y_max = np.max(model_grid_slice[:, 2:4])

    print("Grid slice dimension (X): ", grid_slice_x_min, grid_slice_x_max)
    print("Grid slice dimension (Y): ", grid_slice_y_min, grid_slice_y_max)

    # Remove not-needed columns.
    if slice_dim == 0:
        model_grid_slice_2d = np.delete(model_grid_slice, [0, 1, 6], axis=1)
        section = "x"
    elif slice_dim == 1:
        model_grid_slice_2d = np.delete(model_grid_slice, [2, 3, 6], axis=1)
        section = "y"
    elif slice_dim == 2:
        model_grid_slice_2d = np.delete(model_grid_slice, [4, 5, 6], axis=1)
        section = "z"

    model_final_slice = model_final[slice_filter]

    # ----------------------------------------------------------------------------------
    # Drawing the model.
    # ----------------------------------------------------------------------------------
    grid = model_grid_slice_2d

    """if (draw_true_model):
        draw_model(grid, true_model_slice, "True model.", palette, os.path.join(to_folder, 'true_model.jpg'))"""
    draw_model(
        grid,
        model_final_slice,
        "Final model.",
        palette,
        os.path.join(to_folder, f"final_slice_model_{section}.jpg"),
    )

    # ----------------------------------------------------------------------------------
    # Extract data slice.
    # ----------------------------------------------------------------------------------
    # Select the data located above the model grid slice.
    data_filter_x = np.logical_and(
        data_observed[:, 0] >= grid_slice_x_min, data_observed[:, 0] <= grid_slice_x_max
    )
    data_filter_y = np.logical_and(
        data_observed[:, 1] >= grid_slice_y_min, data_observed[:, 1] <= grid_slice_y_max
    )
    data_filter = np.logical_and(data_filter_x, data_filter_y)

    # data_observed_slice = data_observed[data_filter, :]
    # data_calculated_slice = data_calculated[data_filter, :]

    # print("Ndata slice =", data_observed_slice.shape[0])
    # plot_field(data_filter, 'Observed Data', to_file=os.path.join(to_folder, f'data_observed.jpg'))
    # ----------------------------------------------------------------------------------
    # Drawing the data.
    # ----------------------------------------------------------------------------------
    # Choose coordinate to be used for 1D data plot (data responce from 2D profile).
    if slice_dim == 0:
        # YZ profile.
        profile_coord = 1
    elif slice_dim == 1:
        # XZ profile.
        profile_coord = 0
    else:
        # A 2D data responce - not supported here.
        profile_coord = 0

    # draw_data(data_observed_slice, data_calculated_slice, profile_coord, os.path.join(to_folder, f'data_{section}.jpg'))


def plot_space_delimited_data(filename, to_folder):
    """
    Read a space-delimited file and plot data with specified requirements.

    Parameters:
    filename (str): Path to the space-delimited file

    The function:
    - Skips the first line (header)
    - Uses columns 1, 2, and 4 (0-indexed: 0, 1, 3)
    - Plots x=col1, y=col2, colored by col4
    - Uses viridis colormap with no stroke
    - Clips color scale to 95% of data range
    """

    # Read the file and parse data

    data_observed = np.loadtxt(filename, dtype=float, usecols=(0, 1, 2, 3), skiprows=1)

    # Convert to numpy arrays
    x = np.array(data_observed[:, 0])
    y = np.array(data_observed[:, 1])
    values = np.array(data_observed[:, 3])
    resolution = int(x[1] - x[0])
    # Calculate 95% data range for color clipping

    scatter_to_raster(
        x,
        y,
        values,
        resolution=resolution,
        method="cubic",
        filename=os.path.join(to_folder, "Observed_data.jpg"),
        extent=None,
    )


def scatter_to_raster(
    x,
    y,
    values,
    resolution=100,
    method="linear",
    filename="raster_output.png",
    extent=None,
):
    """
    Convert scatter points to interpolated raster image

    Parameters:
    - x, y: coordinate arrays
    - values: data values at each point
    - resolution: output image resolution
    - method: 'linear', 'nearest', 'cubic'
    - filename: output file path
    - extent: [xmin, xmax, ymin, ymax] or None for auto
    """
    vmin = np.percentile(values, 2.5)  # Lower 2.5%
    vmax = np.percentile(values, 97.5)  # Upper 97.5%

    # Define grid extent
    if extent is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = extent

    # Create regular grid
    xi = np.linspace(xmin, xmax, resolution)
    yi = np.linspace(ymin, ymax, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate
    zi = griddata((x, y), values, (xi_grid, yi_grid), method=method)

    # Save as image
    pl.figure(figsize=(10, 8))
    pl.imshow(
        zi,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equant",
        cmap="viridis",
    )
    pl.colorbar()
    pl.savefig(filename, dpi=300, bbox_inches="tight")
    pl.close()


# Example usage:
# plot_space_delimited_data('your_file.txt')
# =============================================================================
if __name__ == "__main__":
    main()
