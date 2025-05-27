import pygame
import numpy as np
from render_utils import *
from utils import *
import json

# fetch lookup table for rubik's cubes
f = open("./lookup_tables/renderer_posfaceidx.json", "r")
r_pos_face_idx = eval(json.load(f))
f.close()

class Point:
    def __init__(self, pos):
        """Creates 3-dimensional point.
        Args:
            pos: position vector. Points use x=0 and y=0 as the center of the screen"""
        self.pos = np.array(pos)
    def get_draw_coords(self, surface, camera_position):
        """Get the (x,y) coordinates for drawing this point to the screen"""
        width, height = surface.get_width(), surface.get_height()
        z_0 = camera_position[2]
        zp = z_0 - self.z
        return (z_0/zp * self.x + width // 2, z_0/zp * self.y + height // 2)
    def translate(self, delta_pos):
        """Translate this point by tuple or ndarray ```delta_pos```."""
        delta_pos = np.array(delta_pos)
        self.pos += delta_pos
    def transform(self, matrix):
        """Multiply this point's position vector by ```matrix```."""
        self.pos = self.pos @ matrix
    @property
    def x(self):
        return self.pos[0]
    @property
    def y(self):
        return self.pos[1]
    @property
    def z(self):
        return self.pos[2]

class Line:
    def __init__(self, points : list[tuple], color : tuple, thickness):
        """Creates 3-dimensional triangle
        Args:
            points: list of two position tuples
            color: an RGB color tuple
            thickness: thickness of the line"""
        self.points : list[Point] = []
        for coord_tuple in points:
            self.points.append(Point(coord_tuple))
        
        self.color = color
        self.thickness = thickness

    def draw(self, surface, camera_position):
        """Draw this line onto a surface 
        Args:
            surface: the surface to draw onto
            camera_position: camera's position tuple"""
        draw_points = [point.get_draw_coords(surface, camera_position) for point in self.points]
        pygame.draw.polygon(surface, self.color, draw_points, self.thickness)

    def translate(self, delta_pos):
        """Translate this line by tuple ```delta_pos```"""
        for point in self.points:
            point.translate(delta_pos)
    def transform(self, matrix):
        """Multiply this line's position vectors by ```matrix```"""
        for point in self.points:
            point.transform(matrix)

    def midpoint(self):
        return (self.points[0].pos + self.points[1].pos) / 2
    def average_z(self):
        return sum([point.z for point in self.points]) / 2
    def min_z(self):
        return min([point.z for point in self.points])

    def clone(self):
        return Line([point.pos for point in self.points], self.color, self.thickness)


class Triangle:
    def __init__(self, points : list[tuple], color : tuple, flipNormal = False):
        """Creates 3-dimensional triangle
        Args:
            points: list of position tuples
            color: an RGB color tuple
            flipNormal: flips the normal vector when calculating visibility. default=False"""
        self.points : list[Point] = [] # list of Point objects
        for coord_tuple in points:
            self.points.append(Point(coord_tuple))
        
        self.color = color
        self.flipNormal = flipNormal

    def draw(self, surface, camera_position):
        """Draw this triangle onto a surface 
        Args:
            surface: the surface to draw onto
            camera_position: camera's position tuple"""

        # do not draw if normal points wrong way
        P = self.points[1].pos - self.points[0].pos
        Q = self.points[2].pos - self.points[0].pos
        normal = cross(P, Q)
        if self.flipNormal:
            normal = -normal
        if dot(normal, camera_position - self.centroid()) < 0:
            return

        draw_points = [point.get_draw_coords(surface, camera_position) for point in self.points]
        pygame.draw.polygon(surface, self.color, draw_points, 0)

    def translate(self, delta_pos):
        """Translate this triangle by tuple ```delta_pos```"""
        for point in self.points:
            point.translate(delta_pos)
    def transform(self, matrix):
        """Multiply this triangle's position vectors by ```matrix```"""
        for point in self.points:
            point.transform(matrix)

    def centroid(self):
        return (self.points[0].pos + self.points[1].pos + self.points[2].pos) / 3
    def average_z(self):
        return sum([point.z for point in self.points]) / 3
    def min_z(self):
        return min([point.z for point in self.points])

    def clone(self):
        return Triangle([point.pos for point in self.points], self.color, self.flipNormal)


class Cube:
    triangle_positions = np.array([
            [[1, -1, 1], [-1, -1, 1], [-1, -1, -1]],
            [[1, -1, 1], [1, -1, -1], [-1, -1, -1]],
            [[-1, 1, 1], [1, 1, 1], [-1, 1, -1]],
            [[1, 1, -1], [1, 1, 1], [-1, 1, -1]],
            [[1, -1, 1], [1, 1, 1], [-1, -1, 1]],
            [[-1, 1, 1], [1, 1, 1], [-1, -1, 1]],
            [[-1, 1, -1], [1, 1, -1], [-1, -1, -1]],
            [[1, -1, -1], [1, 1, -1], [-1, -1, -1]],
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1]],
            [[-1, 1, 1], [-1, -1, 1], [-1, 1, -1]],
            [[1, 1, 1], [1, -1, 1], [1, 1, -1]],
            [[1, -1, -1], [1, -1, 1], [1, 1, -1]],
        ])
    vertices = np.array([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1]
        ])

    def __init__(self, pos, r, colors, border_color = None, border_thickness = 0, border_offset = 0):
        """Creates a cube.
        Args:
            p: position tuple
            r: half of the side length
            colors: list of Colors applied in UD FB LR order. If a Color is None, doesn't add this face.
            border_color: color of the border, default=None
            border_thickness: thickness of the border. 0 = no border. default=0
            border_offset: how much the border portrudes from the cube. recommended=border_thickness/2 - (small value)"""
        self.triangles = {} # a dictionary of THIS CUBE's triangles 
        self.pos = pos
        self.r = r
        self.colors = colors

        faces = ["U1", "U2", "D1", "D2", "F1", "F2", "B1", "B2", "L1", "L2", "R1", "R2"]
        
        triangle_positions = (r * Cube.triangle_positions).tolist()
        for i in range(12):
            color = colors[i // 2]
            if color is None:
                continue 

            coord_arr = triangle_positions[i]
            flipNormal = i % 2

            triangle = Triangle(coord_arr, color, flipNormal)
            triangle.translate(pos)

            self.triangles[faces[i]] = triangle

        if border_thickness > 0:
            # add borders/edges
            n_edges_added = 0
            vertices = ((r + border_offset) * Cube.vertices).tolist()
            for vertex in vertices:
                """points on a cube can be thought of as bitstrings
                an edge exists between all x,y where x and y have one bit different
                for ordering we also impose x > y, which just means 
                we only search 1 -> 0 bitflips and not 0 -> 1 bitflips"""
                for i in range(3):
                    if vertex[i] > 0:
                        copy = list(vertex)
                        copy[i] = -copy[i]
                        line = Line([vertex, copy], border_color, border_thickness)
                        line.translate(pos)
                        self.triangles[f"EDGE{n_edges_added}"] = line
                        n_edges_added += 1


    def tick(self, dt):
        ...

    def add_tris(self, triangles_list):
        """Adds this Cube's triangles to ```triangles_list```.
        Clones, so it is safe to modify these triangles."""
        for tri in self.triangles.values():
            triangles_list.append(tri.clone())

class RubiksCube:
    color_map = {
        "W" : Color.WHITE,
        "Y" : Color.YELLOW,
        "O" : Color.ORANGE,
        "B" : Color.BLUE,
        "R" : Color.RED,
        "G" : Color.GREEN,
    }

    def __init__(self, r=30, cube_string=solved_string):
        """Create a Rubik's Cube"""
        self.r = r
        self.cubes : list[Cube] = []
        self.set_cube_string(cube_string)

    def add_tris(self, triangles):
        for cube in self.cubes:
            cube.add_tris(triangles)

    def set_cube_string(self, cube_string):
        self.cube_string = cube_string
        for pos, faces in r_pos_face_idx.items():
            # Position tuples are in L/R, U/D, F/B order 
            # Cube init is in U/D, F/B, L/R order
            cube_pos = list(2 * self.r * coord for coord in pos)
            cube_pos[1] = -cube_pos[1] # Correct opposite sign convention for y coordinate
            cube_colors = [None] * 6

            curr_face = 0
            x, y, z = pos

            if x != 0:
                idx = faces[curr_face]
                color = RubiksCube.color_map[cube_string[idx]]
                if x == 1:
                    cube_colors[5] = color
                elif x == -1:
                    cube_colors[4] = color
                curr_face += 1
            if y != 0:
                idx = faces[curr_face]
                color = RubiksCube.color_map[cube_string[idx]]
                if y == 1:
                    cube_colors[0] = color
                elif y == -1:
                    cube_colors[1] = color
                curr_face += 1
            if z != 0:
                idx = faces[curr_face]
                color = RubiksCube.color_map[cube_string[idx]]
                if z == 1:
                    cube_colors[2] = color
                elif z == -1:
                    cube_colors[3] = color
                curr_face += 1
            
            self.cubes.append(Cube(cube_pos, self.r, 
                                   cube_colors, 
                                   border_color = Color.BLACK, 
                                   border_thickness = 3, 
                                   border_offset = 0))
        