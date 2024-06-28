import numpy as np
from synthetic_2d.diffsim.boxes import Box2D, Box3D
from synthetic_2d.diffsim.field import (
    ScalarField,
    VectorField,
    spatial_gradient,
    log,
)
import taichi as ti
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import animation
from phi import flow
from synthetic_2d.diffsim.boxes import convert_cube
from synthetic_2d.envs.converter import GridWorld3D
from synthetic_2d.phiflow.simulation_3d import *
from synthetic_2d.diffsim.walk_simulation import WalkerSimulationStoch3D
from synthetic_2d.phiflow.converter import PointCloudConverter
import scipy.sparse.linalg as spl
import taichi as ti

from synthetic_2d.envs.random_obstacles_3d import (
    RandomObstacles3D,
    GridWorld3D,
)
import scipy.sparse as sp


def test_chat_gpt():
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # Generate example 3D time series data
    np.random.seed(42)
    num_points = 100
    time_values = pd.date_range(
        start="2023-01-01", periods=num_points, freq="D"
    )
    x_values = np.random.rand(num_points)
    y_values = np.random.rand(num_points)
    z_values = np.linspace(0, 1, num_points)

    # Create a DataFrame
    df = pd.DataFrame(
        {"Time": time_values, "X": x_values, "Y": y_values, "Z": z_values}
    )

    # Create a scatter plot with a slider
    fig = go.Figure()

    scatter_trace = go.Scatter3d(
        x=df["X"],
        y=df["Y"],
        z=df["Z"],
        mode="markers",
        marker=dict(size=8, color=df["Z"], colorscale="Viridis", opacity=0.8),
    )

    scatter_trace2 = go.Scatter3d(
        x=df["Y"],
        y=df["X"],
        z=df["Z"],
        mode="markers",
        marker=dict(size=8, color=df["Z"], colorscale="Viridis", opacity=0.8),
    )

    fig.add_trace(scatter_trace)
    fig.add_trace(scatter_trace2)
    # Create slider steps
    slider_steps = []

    for i, timestamp in enumerate(df["Time"]):
        step = dict(
            args=[
                {
                    "x": [df["X"][: i + 1]],
                    "y": [df["Y"][: i + 1]],
                    "z": [df["Z"][: i + 1]],
                },
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            label=str(timestamp),
            method="update",
        )
        slider_steps.append(step)

    # Create slider
    slider = dict(
        active=0,
        steps=slider_steps,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=12), prefix="Time:", visible=True, xanchor="right"
        ),
        transition=dict(duration=300, easing="cubic-in-out"),
    )

    # Update layout
    fig.update_layout(
        sliders=[slider],
        scene=dict(
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
            zaxis=dict(title="Z Axis"),
        ),
        title="3D Time Series with Slider",
    )

    # Show the figure
    fig.show()


def test_chat_gpt():
    import plotly.express as px
    import pandas as pd
    import numpy as np

    # Generate example 3D time series data
    np.random.seed(42)
    num_points = 100
    time_values = pd.date_range(
        start="2023-01-01", periods=num_points, freq="D"
    )
    x_values = np.random.rand(num_points)
    y_values = np.random.rand(num_points)
    z_values = np.linspace(0, 1, num_points)

    # Create a DataFrame
    df = pd.DataFrame(
        {"Time": time_values, "X": x_values, "Y": y_values, "Z": z_values}
    )

    # Create an animated scatter plot with a slider
    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        animation_frame="Time",
        size_max=8,
        color="Z",
        range_color=[0, 1],
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
            zaxis=dict(title="Z Axis"),
        ),
        title="3D Time Series with Slider",
    )

    # Show the figure
    fig.show()


def create_gradient_matrix_2d(nx, ny, dx, dy) -> Tuple[sp.lil_array]:
    # Create 1D finite difference matrices for x and y directions (centered differences)
    Dx = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nx, nx), format="csc") / (
        2 * dx
    )
    Dy = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(ny, ny), format="csc") / (
        2 * dy
    )
    # Apply the boundary conditions
    Dx[0, 0] = -1 / dx
    Dx[0, 1] = 1 / dx
    Dx[-1, -1] = 1 / dx
    Dx[-1, -2] = -1 / dx

    Dy[0, 0] = -1 / dy
    Dy[0, 1] = 1 / dy
    Dy[-1, -1] = 1 / dy
    Dy[-1, -2] = -1 / dy

    # Kronecker product to obtain 2D gradient matrices
    Gx = sp.lil_array(sp.kron(sp.eye(ny), Dx))
    Gy = sp.lil_array(sp.kron(Dy, sp.eye(nx)))
    return Gx, Gy


def create_gradient_matrix_3d(nx, ny, nz, dx, dy, dz) -> Tuple[sp.lil_array]:
    # Create 1D finite difference matrices for x and y directions (centered differences)
    Dx = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nx, nx), format="csc") / (
        2 * dx
    )
    Dy = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(ny, ny), format="csc") / (
        2 * dy
    )
    Dz = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nz, nz), format="csc") / (
        2 * dz
    )
    # Apply the boundary conditions
    Dx[0, 0] = -1 / dx
    Dx[0, 1] = 1 / dx
    Dx[-1, -1] = 1 / dx
    Dx[-1, -2] = -1 / dx

    Dy[0, 0] = -1 / dy
    Dy[0, 1] = 1 / dy
    Dy[-1, -1] = 1 / dy
    Dy[-1, -2] = -1 / dy

    Dz[0, 0] = -1 / dz
    Dz[0, 1] = 1 / dz
    Dz[-1, -1] = 1 / dz
    Dz[-1, -2] = -1 / dz

    # Kronecker product to obtain 2D gradient matrices
    Gx = sp.lil_array(sp.kron(sp.eye(ny * nz), Dx))
    Gy = sp.lil_array(sp.kron(sp.eye(nz), sp.kron(Dy, sp.eye(nx))))
    Gz = sp.lil_array(sp.kron(Dz, sp.eye(nx * ny)))
    return Gx, Gy, Gz


def create_div_matrix_2d(nx, ny, dx, dy) -> sp.lil_matrix:
    Gx, Gy = create_gradient_matrix_2d(nx, ny, dx, dy)
    return sp.hstack((Gx, Gy))


def create_div_matrix_3d(nx, ny, nz, dx, dy, dz) -> sp.lil_matrix:
    Gx, Gy, Gz = create_gradient_matrix_3d(nx, ny, nz, dx, dy, dz)
    return sp.hstack((Gx, Gy, Gz))


def create_laplacian_matrix_2d(
    nx: int, ny: int, dx: float, dy: float
) -> sp.lil_array:
    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    return sp.lil_array(sp.kronsum(Dyy, Dxx))


def create_laplacian_matrix_3d(
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> sp.lil_array:
    # Compute the 2D laplacian matrix
    laplace_2d = create_laplacian_matrix_2d(nx, ny, dx, dy)

    Dzz = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz)) / dz**2
    return sp.lil_array(sp.kronsum(Dzz, laplace_2d))


def get_laplacian(obstacle_map: np.ndarray) -> Tuple[sp.spmatrix, np.ndarray]:
    """Generate the laplacian matrix from an obstacle map

    Args:
        obstacle_map (np.ndarray): binary map where obstacles are 1 and free space is 0
    """
    if obstacle_map.ndim == 2:
        laplace = create_laplacian_matrix_2d(*obstacle_map.shape, 1.0, 1.0)
    elif obstacle_map.ndim == 3:
        laplace = create_laplacian_matrix_3d(
            *obstacle_map.shape, 1.0, 1.0, 1.0
        )
    else:
        raise ValueError("Obstacle map must be 2D or 3D")
    # Keep only the free space in the laplacian
    ind = obstacle_map.flatten() == 0
    return laplace[:, ind][ind, :], ind


def solve_log_concentration(
    obstacle_map: np.ndarray,
    position: np.ndarray,
    C0=1,
    D=1,
    dx=1.0,
    dy=1.0,
) -> np.ndarray:
    """Solve the laplace equation for the log concentration

    Args:
        obstacle_map (np.ndarray): binary map where obstacles are 1 and free space is 0
        source (np.ndarray): source of the concentration
    """
    assert obstacle_map.ndim == 2, "Obstacle map must be 2D"
    nx, ny = obstacle_map.shape
    non_zero_ind = obstacle_map.flatten() == 0
    nb_free_cells = np.sum(non_zero_ind)

    # Create the laplacian and gradient matrices
    laplace = create_laplacian_matrix_2d(nx, ny, dx, dy)  # NxN
    laplace = sp.csc_array(
        laplace[:, non_zero_ind][non_zero_ind, :]
    )  # (N-k)x(N-k)
    Gx, Gy = create_gradient_matrix_2d(nx, ny, dx, dy)
    Gx = Gx[:, non_zero_ind][non_zero_ind, :]  # (N-k)x(N-k)
    Gy = Gy[:, non_zero_ind][non_zero_ind, :]  # (N-k)x(N-k)

    gradient = sp.csc_array(
        sp.vstack((Gx, Gy), format="csc")
    )  # 2(N-k) x (N-k)
    eye_2D = sp.csc_array(
        sp.hstack((sp.eye(nb_free_cells), sp.eye(nb_free_cells)), format="csc")
    )  # (N-k) x 2(N-k)

    # Flatten the source and keep only the free space
    source = np.zeros_like(obstacle_map)
    source[position[0], position[1]] = 1
    source = source.flatten()[non_zero_ind]

    def F(L):
        return laplace @ L + eye_2D @ (gradient @ L) ** 2 + 1 / D * source

    def Jac(L):
        return laplace + 2 * eye_2D @ sp.diags(gradient @ L) @ gradient

    # Initialize the solution
    meshgrid = np.mgrid[0:nx, 0:ny]
    green = (
        -C0
        / (2 * np.pi * D)
        * np.log(
            np.linalg.norm(
                [meshgrid[0] - position[0], meshgrid[1] - position[1]], axis=0
            )
        )
    )
    green[position[0], position[1]] = -C0 / (2 * np.pi * D) * np.log(dx)
    L0 = np.log(green - 1.1 * np.min(green)).flatten()[non_zero_ind]
    return L0, F, Jac


def test_solve_log_concentration():
    import scipy.optimize as so

    obstacle_map = np.zeros((20, 20))
    position = np.array([10, 10])

    L0, F, Jac = solve_log_concentration(obstacle_map, position, D=1e-2, C0=1)
    L = L0.copy()
    for i in range(1):
        delta = spl.spsolve(Jac(L), -F(L))
        # delta, exit_code = spl.cg(Jac(L), -F(L), x0=L, maxiter=1_000)
        # print(exit_code)
        L += delta

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(L0.reshape(obstacle_map.shape), origin="lower")
    ax[1].imshow(F(L0).reshape(obstacle_map.shape), origin="lower")
    ax[2].imshow(Jac(L0).toarray(), origin="lower")
    ax[3].imshow(L.reshape(obstacle_map.shape), origin="lower")

    # solution = so.root(F, L0, jac=Jac, method="lm")
    # print(solution.message)
    # ax[3].imshow(solution.x.reshape(obstacle_map.shape), origin="lower")
    plt.show()


def example_2D(random=False):
    if random:
        obstacle_map = np.random.choice([0, 1], size=(20, 20), p=[0.7, 0.3])
    else:
        obstacle_map = np.zeros((20, 20))
    init_shape = obstacle_map.shape
    laplacian, ind = get_laplacian(obstacle_map)
    grad_x, grad_y = create_gradient_matrix_2d(*obstacle_map.shape, 1.0, 1.0)

    source = np.zeros(laplacian.shape[0])
    source[123] = 1e16
    splu = spl.splu(-laplacian)
    temp_sol = splu.solve(source)

    solution = np.zeros_like(obstacle_map.flatten())
    solution[ind] = temp_sol
    solution = solution.reshape(init_shape)
    log_solution = np.log(solution)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(obstacle_map)
    ax[1].imshow(laplacian.toarray())
    ax[2].imshow(log_solution, origin="lower")

    fig2, ax2 = plt.subplots(2, 2)
    test_x = grad_x @ log_solution.flatten()
    test_y = grad_y @ log_solution.flatten()
    ax2[0, 0].imshow(grad_x.toarray())
    ax2[0, 1].imshow(grad_y.toarray())
    im1 = ax2[1, 0].imshow(test_x.reshape(init_shape))
    im2 = ax2[1, 1].imshow(test_y.reshape(init_shape))
    fig2.colorbar(im1, ax=ax2[1, 0])
    fig2.colorbar(im2, ax=ax2[1, 1])
    ax[2].quiver(
        test_x.reshape(init_shape),
        test_y.reshape(init_shape),
    )

    fig3, ax3 = plt.subplots(1, 3)
    div = create_div_matrix_2d(*obstacle_map.shape, 1.0, 1.0)
    ax3[0].imshow(div.toarray())
    ax3[1].imshow(
        (
            -div
            @ np.concatenate(
                (grad_x @ solution.flatten(), grad_y @ solution.flatten()),
                axis=0,
            ).flatten()
        ).reshape(init_shape)
    )
    ax3[2].imshow(source.reshape(init_shape))
    plt.show()


def example_3D(slice_ind=10, random=False):
    if random:
        obstacle_map = np.random.choice(
            [0, 1], size=(20, 20, 20), p=[0.7, 0.3]
        )
    else:
        obstacle_map = np.zeros((20, 20, 20))
    init_shape = obstacle_map.shape
    laplacian, ind = get_laplacian(obstacle_map)
    grad_x, grad_y, grad_z = create_gradient_matrix_3d(
        *obstacle_map.shape, 1.0, 1.0, 1.0
    )

    source = np.zeros(laplacian.shape[0])
    source[123] = 1e3
    splu = spl.splu(-laplacian)
    temp_sol = splu.solve(source)

    solution = np.zeros_like(obstacle_map.flatten())
    solution[ind] = temp_sol
    solution = solution.reshape(init_shape)
    log_solution = np.log(solution)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(obstacle_map[slice_ind])
    ax[1].imshow(laplacian.toarray())
    ax[2].imshow(log_solution[slice_ind], origin="lower")

    test_x = grad_x @ log_solution.flatten()
    test_y = grad_y @ log_solution.flatten()
    test_z = grad_z @ log_solution.flatten()

    fig2, ax2 = plt.subplots(2, 3)
    ax2[0, 0].imshow(grad_x.toarray())
    ax2[0, 1].imshow(grad_y.toarray())
    ax2[0, 2].imshow(grad_z.toarray())
    im1 = ax2[1, 0].imshow(test_x.reshape(init_shape)[slice_ind])
    im2 = ax2[1, 1].imshow(test_y.reshape(init_shape)[slice_ind])
    im3 = ax2[1, 2].imshow(test_z.reshape(init_shape)[slice_ind])
    fig2.colorbar(im1, ax=ax2[1, 0])
    fig2.colorbar(im2, ax=ax2[1, 1])
    fig2.colorbar(im3, ax=ax2[1, 2])
    ax[2].quiver(
        test_x.reshape(init_shape)[slice_ind],
        test_y.reshape(init_shape)[slice_ind],
    )
    fig.show()
    fig2.show()
    plt.show()


if __name__ == "__main__":
    test_chat_gpt()
    # test_solve_log_concentration()
    # example_2D(random=False)
