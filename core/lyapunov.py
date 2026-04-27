import torch

def estimate_lyapunov(traj):
    """
    Простая оценка максимального показателя Ляпунова
    через рост расстояний между соседними состояниями
    """

    diffs = torch.norm(traj[1:] - traj[:-1], dim=1)

    # избегаем log(0)
    diffs = diffs + 1e-8

    lyap = torch.mean(torch.log(diffs))
    return lyap
