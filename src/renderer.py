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
z_0 = 500
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
    GRAY = (150, 150, 150)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    ORANGE = (255, 165, 0)

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
    def __init__(self, points : list[Point], color : tuple, flipNormal = False):
        """Creates 3-dimensional triangle
        Args:
            points: list of Points
            color: an RGB color tuple"""
        self.points = points
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
        rotation_strength = 2.5
        xnorm = np.sqrt(1/2) * np.tanh(rotation_strength * xnorm)
        ynorm = np.sqrt(1/2) * np.tanh(rotation_strength * ynorm)

        # projection onto sphere of radius 1
        znorm = np.sqrt(1 - xnorm ** 2 - ynorm ** 2)
        return np.array([xnorm, ynorm, znorm])

    def get_matrix(original_pos, new_pos):
        """Compute rotation matrix from original and new mouse position tuples"""
        # get vectors on radius 1 sphere 
        v1 = Arcball.get_arcball_vec(original_pos)
        w1 = Arcball.get_arcball_vec(new_pos)

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

# Create multiple triangles with different positions and colors
objects = [    
    Triangle(
        [Point([75, -75, 75]), 
         Point([75, 75, 75]), 
         Point([-75, -75, 75])],
        Color.GREEN,),
    Triangle(
        [Point([-75, 75, 75]), 
         Point([75, 75, 75]), 
         Point([-75, -75, 75])],
        Color.GREEN,
        flipNormal=True),
    Triangle(
        [Point([-75, -75, -75]), 
         Point([-75, -75, 75]), 
         Point([-75, 75, -75])],
        Color.ORANGE,),
    Triangle(
        [Point([-75, 75, 75]), 
         Point([-75, -75, 75]), 
         Point([-75, 75, -75])],
        Color.ORANGE,
        flipNormal=True),
    Triangle(
        [Point([75, 75, 75]), 
         Point([75, -75, 75]), 
         Point([75, 75, -75])],
        Color.RED,),
    Triangle(
        [Point([75, -75, -75]), 
         Point([75, -75, 75]), 
         Point([75, 75, -75])],
        Color.RED,
        flipNormal=True),
    Triangle(
        [Point([-75, 75, -75]), 
         Point([75, 75, -75]), 
         Point([-75, -75, -75])],
        Color.BLUE,),
    Triangle(
        [Point([75, -75, -75]), 
         Point([75, 75, -75]), 
         Point([-75, -75, -75])],
        Color.BLUE,
        flipNormal=True),
    Triangle(
        [Point([75, -75, 75]), 
         Point([-75, -75, 75]), 
         Point([-75, -75, -75])],
        Color.WHITE,),
    Triangle(
        [Point([75, -75, 75]), 
         Point([75, -75, -75]), 
         Point([-75, -75, -75])],
        Color.WHITE,
        flipNormal=True),
    Triangle(
        [Point([-75, 75, 75]), 
         Point([75, 75, 75]), 
         Point([-75, 75, -75])],
        Color.YELLOW,),
    Triangle(
        [Point([75, 75, -75]), 
         Point([75, 75, 75]), 
         Point([-75, 75, -75])],
        Color.YELLOW,
        flipNormal=True),
]

def translate(delta_pos):
    for object in objects:
        object.translate(delta_pos)
def transform(matrix):
    for object in objects:
        object.transform(matrix)


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
    if keys[pygame.K_w]:
        translate((0, 0, 10))
    if keys[pygame.K_s]:
        translate((0, 0, -10))
    if keys[pygame.K_a]:
        translate((10, 0, 0))
    if keys[pygame.K_d]:
        translate((-10, 0, 0))
    if keys[pygame.K_q]:
        translate((0, 10, 0))
    if keys[pygame.K_e]:
        translate((0, -10, 0))

    if dragging:
        if last_mouse_pos:
            curr_mouse_pos = pygame.mouse.get_pos()
            R = Arcball.get_matrix(last_mouse_pos, curr_mouse_pos)
            transform(R)
        last_mouse_pos = pygame.mouse.get_pos()

    # Fill background
    screen.fill(Color.GRAY)
    
    # Draw all objects
    objects = sorted(objects, key = lambda tri : tri.average_z())
    for object in objects:
        object.draw(screen)
    
    # Update display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(60)

# Quit
pygame.quit()
sys.exit()