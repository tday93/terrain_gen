#!/usr/bin/env python3

import json
import numpy as np
from scipy.spatial import Delaunay

from lloyd import Field
import util


class Atlas:
    '''
    Base class to store map data.
    Does not change data, only stores it and handles saving/loading/initilization
    '''

    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width
        self.sea_level = 0
        self.floor = -1000
        self.ceiling = 2000
        self.points = []
        self.regions = None
        self.vor = None
        self.tri = None
        self.elevs = []
        self.precip = []
        self.flow = []

    def write(self):
        map_dict = {
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "points": self.points,
            "elevs": self.elevs
        }

        with open(self.name + ".json", "w") as fo:
            json.dump(map_dict, fo, indent=2, sort_keys=True)

    def load(self, name):
        with open(name + ".json", "r") as fn:
            map_dict = json.load(fn)

        self.name = map_dict["name"]
        self.height = map_dict["height"]
        self.width = map_dict["width"]
        self.points = map_dict["points"],
        self.elevs = map_dict["elevs"]
        self.init_voronoi()
        self.init_delaunay()

    def init_voronoi(self):
        field = Field(points=np.array(self.points))
        self.vor = field.voronoi

    def init_delaunay(self):
        self.tri = Delaunay(self.points)

    def get_point_neighbors(self, point_index):
        indices, indptr = self.tri.vertex_neighbor_vertices

        return indptr[indices[point_index]:indices[point_index + 1]]

    def get_min_neighbor(self, point_index):
        neighbors = self.get_point_neighbors(point_index)

        neigh_elevs = [self.elevs[i] for i in neighbors]

        return neighbors[np.argmin(neigh_elevs)]

    def dist_2d(self, idx1, idx2):
        p1 = self.points[idx1]
        p2 = self.points[idx2]
        return util.dist_2d(p1[0], p1[1], p2[0], p2[1])

    def __repr__(self):
        return f"<Atlas - Name: {self.name}, Height: {self.height}, Width: {self.width}>"
