import torch

def foam_loss(outputs):
    """
    Расхождение между агентами (GRA-пена)
    """
    loss = 0.0
    n = len(outputs)

    for i in range(n):
        for j in range(i + 1, n):
            loss += torch.mean((outputs[i] - outputs[j])**2)

    return loss
