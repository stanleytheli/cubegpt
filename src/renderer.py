import pygame
import sys
import numpy as np

from arcball import Arcball
from render_utils import *
from objects import *
from utils import *

"""
Extremely rudimentary 3d renderer 

"""

pygame.init()

width, height = 800, 600
z_0 = 600 # z_0 controls the FOV. The camera is positioned at (0, 0, z_0) and looks toward (0, 0, 0).
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Rendering - Rubik's Cube")
Arcball.set_dimensions(width, height)

camera_position = np.array([0, 0, z_0])

#cube_string = solved_string
#cube_string = scrambled(10).cube_string
cube_string = "YBGGWYBGBGWWWOOBRYROOWGYRRGWOWGRBYBRORROBBGYYBGOWYYORW"
rubiks_cube = RubiksCube(cube_string=cube_string,
                         r = 40,)

objects = [
    rubiks_cube,
    ]

R = np.identity(3) # total rotation matrix 
def transform(matrix):
    global R
    R = R @ matrix


# Main game loop
running = True
clock = pygame.time.Clock()

dragging = False
can_move = True
last_mouse_pos = None

def tick(dt):
    # at 60 fps, dt = 0.01667
    global running, dragging, last_mouse_pos, can_move, cube_string

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
        Rp = Arcball.axis_angle_rotmatrix((1, 0, 0), 3*dt)
        transform(Rp)
    if keys[pygame.K_DOWN]:
        Rp = Arcball.axis_angle_rotmatrix((1, 0, 0), -3*dt)
        transform(Rp)
    if keys[pygame.K_LEFT]:
        Rp = Arcball.axis_angle_rotmatrix((0, -1, 0), 3*dt)
        transform(Rp)
    if keys[pygame.K_RIGHT]:
        Rp = Arcball.axis_angle_rotmatrix((0, -1, 0), -3*dt)
        transform(Rp)
    if keys[pygame.K_PAGEUP]:
        Rp = Arcball.axis_angle_rotmatrix((0, 0, 1), -3*dt)
        transform(Rp)
    if keys[pygame.K_PAGEDOWN]:
        Rp = Arcball.axis_angle_rotmatrix((0, 0, 1), 3*dt)
        transform(Rp)

    move = None
    if keys[pygame.K_u]:
        move = "U"
    if keys[pygame.K_d]:
        move = "D"
    if keys[pygame.K_f]:
        move = "F"
    if keys[pygame.K_b]:
        move = "B"
    if keys[pygame.K_l]:
        move = "L"
    if keys[pygame.K_r]:
        move = "R"
    if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) and move is not None:
        move += "'"

    if move is not None and can_move:
        cube_string = rotate_string(cube_string, move)
        rubiks_cube.set_cube_string(cube_string)
        can_move = False

    if move is None:
        can_move = True

    if dragging:
        if last_mouse_pos:
            curr_mouse_pos = pygame.mouse.get_pos()
            Rp = Arcball.get_mouse_drag_rotmatrix(last_mouse_pos, curr_mouse_pos)
            transform(Rp)
        last_mouse_pos = pygame.mouse.get_pos()

def render(screen):
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
    triangles = sorted(triangles, key = lambda tri : tri.min_z())
    for tri in triangles:
        tri.draw(screen, camera_position)


dt = 0
while running:
    # Handle events
    tick(dt)

    # Draw scene
    render(screen)

    # Update display
    pygame.display.flip()
    
    # Control frame rate
    dt = clock.tick(60) / 1000

# Quit
pygame.quit()
sys.exit()