#!/usr/bin/env python3

import json


class Atlas:

    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width
        self.points = []
        self.regions = None

    def write(self):
        map_dict = {
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "points": self.points,
        }

        with open(self.name + ".json", "w") as fo:
            json.dump(map_dict, fo, indent=2, sort_keys=True)

    def load(self, name):
        with open(name + ".json", "r") as fn:
            map_dict = json.load(fn)

        self.name = map_dict["name"]
        self.height = map_dict["height"]
        self.width = map_dict["width"]
        self.points = map_dict["points"]
