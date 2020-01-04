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

    # terrain deformations

    bounded_quad(atlas, 250, 400, 0, 200, -0.05, 0, 200)

    calculate_flows(atlas)

    flow_erosion(atlas, 100)

    # show final plot
    show_tricontour_plot(atlas)
    plt.close('all')

    simple_smooth(atlas, 0.5)

    show_tricontour_plot(atlas)
    plt.close('all')


# Setup Functions
def setup(name, height, width):
    # setup atlas
    atlas = Atlas(name, height, width)
    return atlas


def init_points(atlas, num_points):

    # randomly generate a number of points
    points = [(randrange(0, atlas.width), randrange(0, atlas.height)) for i in range(num_points)]

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

    atlas.elevs = [0 for i in atlas.points]


def init_hydro(atlas):

    atlas.precips = [0.2 for x in atlas.points]
    atlas.flows = [0 for x in atlas.points]


# Terrain Functions

def bounded_quad(atlas, x0, y0, zmin, zmax, a, b, c):

    for i, point in enumerate(atlas.points):
        n = util.dist_2d(x0, y0, point[0], point[1])
        dz = a * n**2 + b * n + c
        dz = max(zmin, dz)
        dz = min(zmax, dz)
        atlas.elevs[i] += dz


def simple_smooth(atlas, smooth_factor):

    dzs = []
    for i, elev in enumerate(atlas.elevs):
        neighbor_elevs = [atlas.elevs[n] for n in atlas.get_point_neighbors(i)]
        mean = (sum(neighbor_elevs) + elev) / (len(neighbor_elevs) + 1)

        dz = (elev - mean) * smooth_factor
        dzs.append(dz)

    atlas.elevs = [elev + dzs[i] for i, elev in enumerate(atlas.elevs)]


def calculate_flows(atlas):

    for i in range(len(atlas.points)):
        precip = atlas.precips[i]
        atlas.flows[i] += precip
        flag = True
        next_point = i

        while flag:
            next_point = atlas.get_min_neighbor(next_point)
            if not next_point:
                flag = False
                break
            if atlas.elevs[next_point] <= atlas.sea_level:
                flag = False
                break
            atlas.flows[next_point] += precip


def flow_erosion(atlas, erosion_factor):

    dzs = []
    for i, flow in enumerate(atlas.flows):
        dz = -(flow * erosion_factor)
        dzs.append(dz)

    apply_delta_z_with_mask(atlas, dzs, get_mask(atlas))


# Utility Functions
def get_mask(atlas):
    return [True for x in atlas.points]


def apply_delta_z_with_mask(atlas, dz, mask):

    for i, elev in enumerate(atlas.elevs):
        if mask[i]:
            atlas.elevs[i] += dz[i]


# Display Functions
def show_tri_plot(atlas):

    points = atlas.points.copy()
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()
    input("PRESS ENTER TO CLOSE PLOT")


def show_vor_plot(atlas):

    field = Field(points=np.array(atlas.points))
    plot = voronoi_plot_2d(field.voronoi, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    plot.show()
    input("PRESS ENTER TO CLOSE PLOT")


def show_tricontour_plot(atlas):

    tri = Triangulation(atlas.points[:, 0], atlas.points[:, 1])

    fig, ax = plt.subplots()

    ax.tricontourf(tri, atlas.elevs)
    ax.tricontour(tri, atlas.elevs, colors="black", linewidths=0.5)

    ax.triplot(tri, color="0.7", linewidth=0.5, alpha=0.5)

    plt.show()
    input("PRESS ENTER TO CLOSE PLOT")


if __name__ == "__main__":
    main("map")
