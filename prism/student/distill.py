"""Eq. (12): the shared forward KL, independent of data and label readers."""


def forward_kl(logits, targets):
    from ..baselines.objectives import kd_soft_loss

    return kd_soft_loss(logits, targets)


def synthetic_distillation_smoke(features, targets, seed=7, steps=3):
    """Test-only linear probe. This is never a native PRISM-D result."""
    import torch
    from ..utils import seed_everything

    seed_everything(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    q = torch.as_tensor(targets, dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], q.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_kl(model(x), q)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
