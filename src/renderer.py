import pygame
import sys
import numpy as np

from arcball import Arcball
from render_utils import *
from objects import *

"""
Extremely rudimentary 3d renderer 

"""

pygame.init()

width, height = 800, 600
z_0 = 400 # z_0 controls the FOV. The camera is positioned at (0, 0, z_0) and looks toward (0, 0, 0).
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Rendering")
Arcball.set_dimensions(width, height)

camera_position = np.array([0, 0, z_0])

objects = [
    Cube((0, 0, 0), 75, 
         [Color.WHITE,
          Color.YELLOW,
          Color.GREEN,
          Color.BLUE,
          Color.ORANGE,
          Color.RED],
          border_color=Color.BLACK,
          border_thickness=4,
          border_offset=1.5),
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

def tick(dt):
    # at 60 fps, dt = 0.01667
    global running, dragging, last_mouse_pos

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
        Rp = Arcball.rotation_matrix((0, 0, 1), (0, -3*dt, 1))
        transform(Rp)
    if keys[pygame.K_DOWN]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (0, 3*dt, 1))
        transform(Rp)
    if keys[pygame.K_LEFT]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (-3*dt, 0, 1))
        transform(Rp)
    if keys[pygame.K_RIGHT]:
        Rp = Arcball.rotation_matrix((0, 0, 1), (3*dt, 0, 1))
        transform(Rp)
    if keys[pygame.K_PAGEUP]:
        Rp = Arcball.rotation_matrix((1, 0, 0), (1, -3*dt, 0))
        transform(Rp)
    if keys[pygame.K_PAGEDOWN]:
        Rp = Arcball.rotation_matrix((1, 0, 0), (1, 3*dt, 0))
        transform(Rp)

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