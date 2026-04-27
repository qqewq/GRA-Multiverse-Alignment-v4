import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from core.lorenz import generate_lorenz_trajectory
from core.lyapunov import estimate_lyapunov
from core.models import Agent
from core.gra_loss import foam_loss

# =========================
# 1. Генерация хаоса (Lorenz)
# =========================

traj = generate_lorenz_trajectory(steps=2000)

# =========================
# 2. Оценка Ляпунова
# =========================

lyap = estimate_lyapunov(traj)
print("Estimated Lyapunov exponent:", lyap.item())

# =========================
# 3. Multi-agent GRA alignment
# =========================

agents = [Agent() for _ in range(3)]

optimizer = optim.Adam(
    [p for a in agents for p in a.parameters()],
    lr=0.005
)

loss_history = []

for step in range(500):

    outputs = [agent(traj) for agent in agents]

    loss = foam_loss(outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# =========================
# 4. Визуализация
# =========================

# Lorenz attractor
plt.figure()
plt.plot(traj[:, 0].numpy(), traj[:, 1].numpy())
plt.title("Lorenz attractor (XY)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("lorenz.png")

# Loss curve
plt.figure()
plt.plot(loss_history)
plt.title("GRA Alignment Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("loss.png")

print("Saved: lorenz.png, loss.png")
