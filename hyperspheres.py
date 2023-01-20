# %% 
import numpy as np
import matplotlib.pyplot as plt
from skspatial.objects import Sphere, Points, Line, Cylinder
import torch as t
import math

fig = plt.figure()
ax = fig.add_subplot(111, aspect="equal")

# plot a circle of radius r
r = 1
c1 = plt.Circle((0, 0), r, color="b", fill=False)
ax.add_patch(c1)

# plot a circle of radius epsilon centered on a random point on the circle
epsilon = 1.5


# get a random point on the circle
theta_0 = np.random.uniform(0, 2 * np.pi)
x = r * np.cos(theta_0)
y = r * np.sin(theta_0)

# plot the circle
c2 = plt.Circle((x, y), epsilon, color="r", fill=False)
ax.add_patch(c2)

# Plot the vector from the center to the point
ax.plot([0, x], [0, y], color="black")

# First plot the tangent plane

# Get the normal vector to the tangent plane
normal = np.array([x, y])
normal = normal / np.linalg.norm(normal)

tangent = np.array([-normal[1], normal[0]])

theta = np.arccos(1 - epsilon**2 / (2 * r**2))

epsilon_ext = r * np.sin(theta) + np.tan(theta) * (r - r * np.cos(theta))

# Get the point on the tangent plane
tangent_pt_1 = np.array([x, y]) - epsilon_ext * tangent
tangent_pt_2 = np.array([x, y]) + epsilon_ext * tangent

# Plot the tangent plane
# ax.plot([tangent_pt_1[0], tangent_pt_2[0]], [tangent_pt_1[1], tangent_pt_2[1]], color='black')

# Plotlines to the tangent plane


# Randomly sample a point on the intersection of the tangent plane and the cone of inner angle theta

x = tangent_pt_2 if np.random.uniform() < 0.5 else tangent_pt_1

r_prime = r * np.cos(theta)
x_prime = r_prime * np.cos(theta_0)
y_prime = r_prime * np.sin(theta_0)

ax.plot(x_prime, y_prime, "o", color="black")

tangent_pt_3 = np.array([x_prime, y_prime]) - r * np.sin(theta) * tangent
tangent_pt_4 = np.array([x_prime, y_prime]) + r * np.sin(theta) * tangent

ax.plot(
    [tangent_pt_3[0], tangent_pt_4[0]],
    [tangent_pt_3[1], tangent_pt_4[1]],
    color="black",
)

ax.plot([0, tangent_pt_3[0]], [0, tangent_pt_3[1]], color="black")
ax.plot([0, tangent_pt_4[0]], [0, tangent_pt_4[1]], color="black")

# Plot the point
# ax.plot(x[0], x[1], 'o', color='black')

# Rescale this point to be on the circle
x = x / np.linalg.norm(x) * r

# Plot the point
ax.plot(x[0], x[1], "o", color="black")
# %%

# Generalize to 3D
from skspatial.objects import *

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# plot a sphere of radius r
r = 1.0


def plot_sphere(r, ax, center=[0, 0, 0.0], alpha=0.2):
    sphere = Sphere(center, r)
    sphere.plot_3d(ax, alpha=alpha)


plot_sphere(1, ax, alpha=0.05)

# plot a sphere of radius epsilon centered on a random point on the circle
epsilon = 0.8

# get a random point on the circle (in the quadrant facing uss)
theta = 0  # np.random.uniform(np.pi, np.pi * 3 / 2)
phi = 0  # np.random.uniform(0, np.pi / 2)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

center = np.array([x, y, z])

# plot the circle
plot_sphere(epsilon, ax, center=center, alpha=0.1)

# plot the vector from the center to the point

l = Line(point=[0, 0, 0], direction=[x, y, z])
l.plot_3d(ax, color="black")


# theta is the angle of the cone which goes through the center of the sphere and the intersection

cone_angle = np.arccos(1 - epsilon**2 / (2 * r**2))


# Plot the tangent plane at a distance r_prime

r_prime = r * np.cos(cone_angle)

# Plot a point at the distance r_prime along r

x_prime = r_prime * np.sin(phi) * np.cos(theta)
y_prime = r_prime * np.sin(phi) * np.sin(theta)
z_prime = r_prime * np.cos(phi)

pt_prime = np.array([x_prime, y_prime, z_prime])

Points([[x_prime, y_prime, z_prime]]).plot_3d(ax, color="black")

# Plot a circle of radius epsilon_prime

epsilon_prime = r * np.sin(cone_angle)

cylinder = Cylinder(point=[0, 0, 0], vector=pt_prime, radius=epsilon_prime)
cylinder.plot_3d(ax, alpha=0.2)

# Sample random points on the intersection

# Get a random point on the circle

pts = []

for i in range(100):
    theta_sample = np.random.uniform(0, np.pi * 2)

    x_sample = epsilon_prime * np.cos(theta_sample)
    y_sample = epsilon_prime * np.sin(theta_sample)
    z_sample = 0

    pt_sample = np.array([x_sample, y_sample, z_sample])

    # Rotate the point to be on the tangent plane

    # Get the normal vector to the tangent plane
    normal = np.array([x, y, z])
    normal = normal / np.linalg.norm(normal)

    tangent = np.array([-normal[1], normal[0], 0])

    # Get the rotation matrix

    def get_rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        return np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )

    rotation_matrix = get_rotation_matrix(normal, cone_angle)

    # Shift the point to be on the tangent plane

    pt_sample = pt_sample + pt_prime

    # Rotate the point

    pts.append(np.matmul(rotation_matrix, pt_sample))

# Plot the rotated point

Points(pts).plot_3d(ax, color="black")

# %%

def get_householder_matrix(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """https://math.stackexchange.com/a/4524336/914272"""
    nx = x / t.norm(x)
    ny = y / t.norm(y)
    c = (nx + ny).view(-1)

    return (2 * t.outer(c, c) / (c.T @ c)) - t.eye(c.shape[0])



def apply_householder_matrix_from_vertical_(x: t.Tensor, vs: t.Tensor):
    """Applies the above for the case that y = (0, ..., 0, 1)   
    without having to compute the matrix explicitly.

    The Householder matrix maps a vector |x> onto |y>

    First, you normalize both vectors.
    
    Then you define |c> = |x> + |y>

    Then you define the Householder matrix:
    
    H = 2 |c><c| / (<c|c>) - I

    That is:

    H|v> = 2 |c><c|v> / <c|c> - |v>

    Parameters
    ----------
    x : t.Tensor shape (d,)    
        Vector which defines the rotation. This is where (0, ..., 0, 1) is mapped.
    vs : t.Tensor shape (n, d)
        n Vectors to rotate

    """
    x_norm = t.norm(x)
    x /= x_norm
    x[-1] += 1
    
    vs -= (2 * t.inner(x, vs).view(-1, 1) / t.dot(x, x)) * x.view(1, -1)
    x[-1] -= 1
    x *= x_norm

# Generaalaize to arbitrary dimensions

def sample_from_hypersphere_intersection(
    r: t.Tensor,
    epsilon: float,
    n_samples: int,
):
    """Sample points from the intersection of two hyperspheres.

    Parameters
    ----------
    r : np.ndarray
        Vector that determines the radius of the larger hypersphere
    epsilon : float
        Radius of the smaller hypersphere, centered at r
    n_samples : int
        Number of samples to take

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, r.shape[0])
    """

    d = r.shape[0]
    r_norm = t.norm(r)

    # Get the angle of the cone which goes through the center of the sphere and the intersection
    cone_angle = t.arccos(1 - epsilon**2 / (2 * r_norm**2))

    # Get the perp distance from r to the intersection
    epsilon_inner = r_norm * t.sin(cone_angle)

    # Sample a perturbation from the d-1 dimensional hypersphere of intersection
    perturbations = t.empty(n_samples, d)
    t.nn.init.normal_(perturbations)
    perturbations *= epsilon_inner / t.norm(perturbations[:, :-1], dim=1, keepdim=True)
    perturbations[:, -1] = 0

    # Apply the rotation 
    apply_householder_matrix_from_vertical_(r, perturbations)

    # Shift the perturbations
    perturbations += r * t.cos(cone_angle)

    return perturbations


def sample_from_hypersphere_intersection_2(
    r: t.Tensor,
    epsilon: float,
    n_samples: int,
):
    """Sample points from the intersection of two hyperspheres.

    Parameters
    ----------
    r : np.ndarray
        Vector that determines the radius of the larger hypersphere
    epsilon : float
        Radius of the smaller hypersphere, centered at r
    n_samples : int
        Number of samples to take

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, r.shape[0])
    """

    d = r.shape[0]
    r_norm = t.norm(r)

    # Get the angle of the cone which goes through the center of the sphere and the intersection
    cone_angle = t.arccos(1 - epsilon**2 / (2 * r_norm**2))

    # Get the perp distance from r to the intersection
    epsilon_inner = r_norm * t.sin(cone_angle)

    # Sample a perturbation from the d-1 dimensional hypersphere of intersection
    perturbations = t.empty(n_samples, d)
    t.nn.init.normal_(perturbations)
    perturbations *= epsilon_inner / t.norm(perturbations[:, :-1], dim=1, keepdim=True)
    perturbations[:, -1] = 0

    # Apply the rotation 
    z = t.zeros(d) * 1.
    z[-1] = 1.
    H = get_householder_matrix(z, r) 

    perturbations =  (H @ perturbations.T).T

    # Shift the perturbations
    perturbations += r * t.cos(cone_angle)

    return perturbations


# Plot to confirm

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot the sphere
r = np.array([1.0, 0, 0])
radius = np.linalg.norm(r)
epsilon = 1.9

s1 = Sphere(point=[0, 0, 0], radius=radius)
s1.plot_3d(ax, alpha=0.1)

s2 = Sphere(point=r, radius=epsilon)
s2.plot_3d(ax, alpha=0.1)

perturbations = sample_from_hypersphere_intersection(t.Tensor(r), epsilon, 100).numpy()
pts = Points(perturbations)
pts.plot_3d(ax, color="black")

perturbations_2 = sample_from_hypersphere_intersection_2(t.Tensor(r), epsilon, 100).numpy()
pts_2 = Points(perturbations_2)
pts_2.plot_3d(ax, color="red")

theta = np.arccos(1 - epsilon**2 / (2 * np.linalg.norm(r) ** 2))
r_inner = r * np.cos(theta)
r_prime = r / r_inner

pts_on_line = Points([r, r_inner])
pts_on_line.plot_3d(ax, color="red")

print(
    np.linalg.norm(r),
    np.linalg.norm(perturbations, axis=1),
)

# %%
