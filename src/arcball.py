import numpy as np
from render_utils import *

width = None
height = None

class Arcball:
    def set_dimensions(w, h):
        global width, height
        width = w
        height = h

    def get_arcball_vec(pos):
        """Returns the arcball vector given a mouse position vector"""
        # center coordinates
        x = pos[0] - width // 2
        y = pos[1] - height // 2

        # projection onto circle of radius srqt2 (to contain the whole screen)
        xnorm = 2 * x / width
        ynorm = 2 * y / height
        

        # projection onto circle of radius 1
        r = np.sqrt(xnorm ** 2 + ynorm ** 2) # 0 to sqrt2
        if r > 0: # numerical stability
            rotation_strength = 3.5
            rp = np.sqrt(1/2) * np.tanh(rotation_strength * r) # 0 to ~1
            xnorm = rp/r * xnorm
            ynorm = rp/r * ynorm

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

    def axis_angle_rotmatrix(u, theta):
        """Return the rotation matrix that rotates angle theta about axis u. 
        If u is taken to point out of the page, a positive-theta rotation is clockwise.
        (Would be counterclockwise in a right-handed coordinate system)"""
        u = normalized(np.array(u))
        
        # 'origin' vector
        a = np.array([1, 0, 0])
        a = a - dot(a, u) * u

        epsilon = 0.0001 # take care of the case where a lied entirely along u, making orthogonal gen awkward
        if dot(a, a) < epsilon:
            a = np.array([0, 1, 0])
            a = a - dot(a, u) * u
        
        a = normalized(a)

        # second basis vector
        a2 = cross(u, a)

        # 'target' vector
        b = np.cos(theta) * a + np.sin(theta) * a2

        return Arcball.rotation_matrix(a, b)


    def get_mouse_drag_rotmatrix(original_pos, new_pos):
        """Compute rotation matrix from original and new mouse position tuples"""
        # get vectors on radius 1 sphere 
        v1 = Arcball.get_arcball_vec(original_pos)
        w1 = Arcball.get_arcball_vec(new_pos)
        return Arcball.rotation_matrix(v1, w1)

