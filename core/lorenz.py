import torch

def lorenz_step(state, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    new_state = state + dt * torch.tensor([dx, dy, dz])
    return new_state


def generate_lorenz_trajectory(steps=2000):
    state = torch.tensor([1.0, 1.0, 1.0])
    traj = []

    for _ in range(steps):
        state = lorenz_step(state)
        traj.append(state)

    return torch.stack(traj)
