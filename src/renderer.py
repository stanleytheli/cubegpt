import pygame
import sys
import numpy as np

"""
Extremely rudimentary 3d renderer 

"""

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Rendering")

# z_0 controls the FOV. The camera is positioned at (0, 0, z_0) and looks toward (0, 0, 0).
z_0 = 400
camera_position = np.array([0, 0, z_0])

# Vector utility
def dot(a, b):
    """Scalar dot product of vectors a and b"""
    return np.sum(a * b)
def cross(a, b):
    """Cross product of vectors a and b"""
    return np.cross(a, b)
def normalized(a):
    """Normalize a vector"""
    return a / np.sqrt(dot(a, a))

class Color:
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    WHITE = (245, 245, 245)
    RED = (180, 0, 0)
    GREEN = (0, 180, 0)
    BLUE = (0, 0, 180)
    YELLOW = (255, 240, 7)
    PURPLE = (128, 0, 128)
    ORANGE = (255, 128, 0)

class Point:
    def __init__(self, pos):
        """Creates 3-dimensional point.
        Args:
            pos: position vector. Points use x=0 and y=0 as the center of the screen"""
        self.pos = np.array(pos)
    def get_draw_coords(self):
        """Get the (x,y) coordinates for drawing this point to the screen"""
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

class Triangle:
    def __init__(self, points : list[tuple], color : tuple, flipNormal = False):
        """Creates 3-dimensional triangle
        Args:
            points: list of position tuples
            color: an RGB color tuple"""
        self.points_list = points # for cloning

        self.points = [] # list of Point objects
        for coord_tuple in points:
            self.points.append(Point(coord_tuple))
        
        self.color = color
        self.flipNormal = flipNormal

    def draw(self, surface):
        """Draw this triangle onto a surface"""

        # do not draw if normal points wrong way
        P = self.points[1].pos - self.points[0].pos
        Q = self.points[2].pos - self.points[0].pos
        normal = cross(P, Q)
        if self.flipNormal:
            normal = -normal
        if dot(normal, camera_position - self.centroid()) < 0:
            return

        draw_points = [point.get_draw_coords() for point in self.points]
        pygame.draw.polygon(surface, self.color, draw_points)

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

    def clone(self):
        return Triangle(self.points_list, self.color, self.flipNormal)

class Cube:
    def __init__(self, pos, r, colors):
        """Creates a cube.
        Args:
            p: position tuple
            r: half of the side length
            colors: list of Colors applied in UD FB LR order"""
        self.triangles = {} # a dictionary of THIS CUBE's triangles 
        self.pos = pos
        self.r = r
        self.colors = colors

        triangle_positions = [
            [[r, -r, r], [-r, -r, r], [-r, -r, -r]],
            [[r, -r, r], [r, -r, -r], [-r, -r, -r]],
            [[-r, r, r], [r, r, r], [-r, r, -r]],
            [[r, r, -r], [r, r, r], [-r, r, -r]],
            [[r, -r, r], [r, r, r], [-r, -r, r]],
            [[-r, r, r], [r, r, r], [-r, -r, r]],
            [[-r, r, -r], [r, r, -r], [-r, -r, -r]],
            [[r, -r, -r], [r, r, -r], [-r, -r, -r]],
            [[-r, -r, -r], [-r, -r, r], [-r, r, -r]],
            [[-r, r, r], [-r, -r, r], [-r, r, -r]],
            [[r, r, r], [r, -r, r], [r, r, -r]],
            [[r, -r, -r], [r, -r, r], [r, r, -r]],
        ]
        faces = ["U1", "U2", "D1", "D2", "F1", "F2", "B1", "B2", "L1", "L2", "R1", "R2"]
        for i in range(12):
            coord_arr = triangle_positions[i]
            flipNormal = i % 2
            color = colors[i // 2]

            triangle = Triangle(coord_arr, color, flipNormal)
            triangle.translate(pos)

            self.triangles[faces[i]] = triangle

    def tick(self):
        ...

    def add_tris(self, triangles_list):
        """Adds this Cube's triangles to ```triangles_list```.
        Clones, so it is safe to modify these triangles."""
        for tri in self.triangles.values():
            triangles_list.append(tri.clone())

class Arcball:
    def get_arcball_vec(pos):
        """Returns the arcball vector given a mouse position vector"""
        # center coordinates
        x = pos[0] - width // 2
        y = pos[1] - height // 2

        # projection onto circle of radius srqt2 (to contain the whole screen)
        xnorm = 2 * x / width
        ynorm = 2 * y / height

        # projection onto circle of radius 1
        rotation_strength = 3
        xnorm = np.sqrt(1/2) * np.tanh(rotation_strength * xnorm)
        ynorm = np.sqrt(1/2) * np.tanh(rotation_strength * ynorm)

        # projection onto sphere of radius 1
        znorm = np.sqrt(1 - xnorm ** 2 - ynorm ** 2)
        return np.array([xnorm, ynorm, znorm])

    def rotation_matrix(a, b):
        """Compute the 3d rotation matrix R such that aR = b, when a and b are normalized.
        Since there are multiple rotation matrices that could do this,
        R is chosen so that all rotation is in the direction of (a cross b)"""
        # first basis vectors
        v1 = normalized(np.array(a))
        w1 = normalized(np.array(b))

        # calculate second basis vectors
        delta = w1 - v1

        epsilon = 0.0001 # ignore very small rotations (to avoid division by zero)
        if dot(delta, delta) < epsilon:  
            return np.identity(3)
        
        v2 = delta - dot(delta, v1) * v1
        v2 = normalized(v2)
        w2 = delta - dot(delta, w1) * w1
        w2 = normalized(w2)

        # calculate third basis vectors
        v3 = cross(v1, v2)
        w3 = cross(w1, w2)

        # i = (i dot v1)v1 + (i dot v2)v2 + (i dot v3)v3
        # ==> 
        # i' = (i dot v1)w1 + (i dot v2)w2 + (i dot v3)w3
        i, j, k = np.identity(3)
        ip = dot(i, v1) * w1 + dot(i, v2) * w2 + dot(i, v3) * w3
        jp = dot(j, v1) * w1 + dot(j, v2) * w2 + dot(j, v3) * w3
        kp = dot(k, v1) * w1 + dot(k, v2) * w2 + dot(k, v3) * w3

        R = np.array([ip, jp, kp])
        return R


    def get_mouse_drag_rotmatrix(original_pos, new_pos):
        """Compute rotation matrix from original and new mouse position tuples"""
        # get vectors on radius 1 sphere 
        v1 = Arcball.get_arcball_vec(original_pos)
        w1 = Arcball.get_arcball_vec(new_pos)
        return Arcball.rotation_matrix(v1, w1)

objects = [
    Cube((0, 0, 0), 75, 
         [Color.WHITE,
          Color.YELLOW,
          Color.GREEN,
          Color.BLUE,
          Color.ORANGE,
          Color.RED])
    ]

R = np.identity(3) # total rotation matrix 
def transform(matrix):
    global R
    R = R @ matrix

# Main game loop
running = True
clock = pygame.time.Clock()

dragging = False
last_mouse_pos = None

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            last_mouse_pos = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (0, -0.05, 1))
        transform(Rp)
    if keys[pygame.K_DOWN]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (0, 0.05, 1))
        transform(Rp)
    if keys[pygame.K_LEFT]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (-0.05, 0, 1))
        transform(Rp)
    if keys[pygame.K_RIGHT]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (0.05, 0, 1))
        transform(Rp)
    if keys[pygame.K_PAGEUP]:
        Rp = Arcball.rotation_matrix((1, 0, 0), (1, -0.05, 0))
        transform(Rp)
    if keys[pygame.K_PAGEDOWN]:
        Rp = Arcball.rotation_matrix((1, 0, 0), (1, 0.05, 0))
        transform(Rp)

    if dragging:
        if last_mouse_pos:
            curr_mouse_pos = pygame.mouse.get_pos()
            Rp = Arcball.get_mouse_drag_rotmatrix(last_mouse_pos, curr_mouse_pos)
            transform(Rp)
        last_mouse_pos = pygame.mouse.get_pos()

    # Fill background
    screen.fill(Color.GRAY)
    
    # Add scene's triangles
    triangles = []
    for object in objects:
        object.add_tris(triangles)

    # Apply rotations
    for tri in triangles:
        tri.transform(R)

    # Draw all triangles
    triangles = sorted(triangles, key = lambda tri : tri.average_z())
    for tri in triangles:
        tri.draw(screen)
    
    # Update display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(60)

# Quit
pygame.quit()
sys.exit()