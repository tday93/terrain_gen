#!/usr/bin/env python3
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import voronoi_plot_2d, Delaunay

from lloyd import Field
from atlas import Atlas
import util


def main(name):

    ###############
    # Basic Setup #
    ###############

    # setup(name, height, width)
    atlas = setup(name, 500, 800)
    print(atlas)

    init_points(atlas, 2000)  # Atlas and number of points to generate

    # relax points using lloyd's algorithm, rescale at each stage
    relax_rescale_n(atlas, 6)

    # initilize final voronoi
    atlas.init_voronoi()

    # initialize delaunay triangulation
    atlas.init_delaunay()

    # show voronoi plot
    show_vor_plot(atlas)
    plt.close('all')

    # show delaunay plot
    show_tri_plot(atlas)
    plt.close('all')

    # initialize elevations
    init_elevations(atlas)

    # initialize precip and flow
    init_hydro(atlas)

    ########################
    # Terrain Deformations #
    ########################

    # flat_locus(atlas, x0, y0, r, z):
    flat_locus(atlas, 400, 250, 200, 500)

    # bounded_quad(atlas, x0, y0, zmin, zmax, a, b, c):
    # bounded_quad(atlas, 400, 250, 0, 5000, -0.04, 0, 3000)

    # rand_bounded_quad(atlas, zmin, zmax, a, b, c):
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)
    rand_bounded_quad(atlas, 0, 6000, -0.05, 0, 800)

    show_tricontour_plot(atlas, name="Basic Deformations")
    plt.close('all')

    atlas.calculate_all()

    flow_erosion(atlas, 50)
    atlas.calculate_all()
    flow_erosion(atlas, 50)
    atlas.calculate_all()
    flow_erosion(atlas, 50)
    atlas.calculate_all()
    flow_erosion(atlas, 50)
    atlas.calculate_all()
    flow_erosion(atlas, 50)
    atlas.calculate_all()

    # show final plot
    show_tricontour_plot(atlas, name="Post Erosion, Pre-Smoothing")
    plt.close('all')

    simple_smooth(atlas, 0.5)

    show_tricontour_plot(atlas, name="Post-Smoothing")
    plt.close('all')


# Setup Functions
def setup(name, height, width):
    # setup atlas
    atlas = Atlas(name, height, width)
    return atlas


def init_points(atlas, num_points):

    # randomly generate a number of points
    points = [(randrange(0, atlas.width),
               randrange(0, atlas.height))
              for i in range(num_points)]

    # add bounding points
    points.append([0, 0])
    points.append([atlas.width, 0])
    points.append([0, atlas.height])
    points.append([atlas.width, atlas.height])

    atlas.points = np.array(points)


def relax_rescale_n(atlas, n):

    for i in range(n):
        print(f"Relaxation iteration {i}")
        relax_points(atlas)
        rescale_points(atlas)


def relax_points(atlas):

    field = Field(points=np.array(atlas.points))
    field.relax_points()
    atlas.points = field.points
    atlas.vor = field.voronoi


def rescale_points(atlas):

    points = np.array(atlas.points)
    x = points[:, 0]
    y = points[:, 1]
    bb = [min(x), max(x), min(y), max(y)]

    scaled_points = []

    for point in atlas.points:
        new_x = ((point[0] - bb[0]) / (bb[1] - bb[0])) * atlas.width
        new_y = ((point[1] - bb[2]) / (bb[3] - bb[2])) * atlas.height

        scaled_points.append([new_x, new_y])

    atlas.points = np.array(scaled_points)


def init_elevations(atlas):

    atlas.elevs = [-50 for i in atlas.points]


def init_hydro(atlas):

    atlas.precips = [0.2 for x in atlas.points]
    atlas.flows = [0 for x in atlas.points]


# Terrain deformation functions
def bounded_quad(atlas, x0, y0, zmin, zmax, a, b, c):

    for i, point in enumerate(atlas.points):
        n = util.dist_2d(x0, y0, point[0], point[1])
        dz = a * n**2 + b * n + c
        dz = max(zmin, dz)
        dz = min(zmax, dz)
        atlas.elevs[i] += dz


def rand_bounded_quad(atlas, zmin, zmax, a, b, c):
    x0 = randrange(0, atlas.width)
    y0 = randrange(0, atlas.height)

    for i, point in enumerate(atlas.points):
        n = util.dist_2d(x0, y0, point[0], point[1])
        dz = a * n**2 + b * n + c
        dz = max(zmin, dz)
        dz = min(zmax, dz)
        atlas.elevs[i] += dz


def flat_locus(atlas, x0, y0, r, z):

    for i, point in enumerate(atlas.points):
        dz = 0
        n = util.dist_2d(x0, y0, point[0], point[1])
        if (n <= r):
            dz = z
        atlas.elevs[i] += dz


def simple_smooth(atlas, smooth_factor):

    dzs = []
    for i, elev in enumerate(atlas.elevs):
        neighbor_elevs = [atlas.elevs[n] for n in atlas.get_point_neighbors(i)]
        mean = (sum(neighbor_elevs) + elev) / (len(neighbor_elevs) + 1)

        dz = (mean - elev) * smooth_factor
        dzs.append(dz)

    atlas.elevs = [elev + dzs[i] for i, elev in enumerate(atlas.elevs)]


def flow_erosion(atlas, erosion_factor):

    mask = get_sea_level_mask(atlas)
    dzs = []
    for i, flow in enumerate(atlas.flows):
        dz_neigh = atlas.get_min_neighbor_height(i) - atlas.elevs[i]
        if (atlas.flow_vs[i] == 0):
            dz = flow * erosion_factor
            dz = min(dz, dz_neigh)
        else:
            dz = -(flow * erosion_factor)
            dz = max(dz, dz_neigh)

        dzs.append(dz)

    apply_delta_z_with_mask_bounded(atlas, dzs, mask, 0, 8800)


# Utility Functions
def get_mask(atlas):
    return [True for x in atlas.points]


def get_sea_level_mask(atlas):
    return [True if atlas.elevs[i] > atlas.sea_level
            else False for i, point in enumerate(atlas.points)]


def apply_delta_z_with_mask(atlas, dz, mask):

    for i, elev in enumerate(atlas.elevs):
        if mask[i]:
            atlas.elevs[i] += dz[i]


def apply_delta_z_with_mask_bounded(atlas, dz, mask, z_min, z_max):

    for i, elev in enumerate(atlas.elevs):
        if mask[i]:
            new_z = atlas.elevs[i] + dz[i]
            new_z = min(z_max, new_z)
            new_z = max(z_min, new_z)
            atlas.elevs[i] = new_z


# Display Functions
def show_tri_plot(atlas):

    points = atlas.points.copy()
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()
    input("Press enter for next plot")


def show_vor_plot(atlas):

    field = Field(points=np.array(atlas.points))
    plot = voronoi_plot_2d(field.voronoi, show_vertices=False,
                           line_colors='orange', line_width=2,
                           line_alpha=0.6, point_size=2)
    plot.show()
    input("Press enter for next plot")


def show_tricontour_plot(atlas, name="Default Name"):

    tri = Triangulation(atlas.points[:, 0], atlas.points[:, 1])

    fig, ax = plt.subplots()

    ax.tricontourf(tri, atlas.elevs,
                   levels=[-3000, -2000, -1000, -500, -100,
                           0, 100, 500, 1000, 2000, 3000, 4000,
                           5000, 6000, 7000, 8000, 8500],
                   colors=["#71abd8ff", "#84b9e3ff", "#a1d2f7ff", "#c6ecffff",
                           "#d8f2feff", "#acd0a5ff", "#94bf8bff", "#bdcc96ff",
                           "#d1d7abff", "#efebc0ff", "#ded6a3ff", "#cab982ff",
                           "#b9985aff", "#ac9a7cff", "#cac3b8ff", "#f5f4f2ff"])

    ax.tricontour(tri, atlas.elevs,
                  levels=[-3000, -2000, -1000, -500, -100,
                          0, 100, 500, 1000, 2000, 3000, 4000,
                          5000, 6000, 7000, 8000, 8500],
                  linestyles=["solid", "solid", "solid",
                              "dashed", "solid", "dashdot",
                              "dashed", "solid", "solid",
                              "solid", "solid", "solid",
                              "solid", "solid", "solid",
                              "dashed"],
                  colors="black", linewidths=0.5)

    ax.triplot(tri, color="0.7", linewidth=0.5, alpha=0.5)

    fig.canvas.manager.set_window_title(name)

    plt.show()
    input("Press enter for next plot")


if __name__ == "__main__":
    main("map")
