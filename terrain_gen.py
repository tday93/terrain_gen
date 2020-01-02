#!/usr/bin/env python3
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d, Delaunay


from lloyd import Field
from atlas import Atlas


def main(name):

    atlas = setup(name, 500, 800)
    print(atlas)

    init_points(atlas, 1000)  # Atlas and number of points to generate

    relax_rescale_n(atlas, 6)

    init_delaunay(atlas)

    show_vor_plot(atlas)
    plt.close('all')
    show_tri_plot(atlas)
    plt.close('all')


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
        print(f"Relaxation iteration {n}")
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


def init_delaunay(atlas):

    points = atlas.points.copy()
    tri = Delaunay(points)
    atlas.tri = tri


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


if __name__ == "__main__":
    main("map")
