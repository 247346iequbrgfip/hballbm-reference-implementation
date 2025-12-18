import torch


def latent_relevance_over_time(model, x_img, x_cli, x_gen, y, timesteps):
    """
    Computes a simple relevance proxy on the latent trajectory:
      R(t) = h(t) * d(score)/d(h(t))

    Returns:
      R: [T, N, D]
      H: [T, N, D] 
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    # Forward with trajectory grads enabled
    logits, H = model(x_img, x_cli, x_gen, timesteps, retain_trajectory_grads=True)
  
    score = logits.gather(1, y.view(-1, 1)).sum()
    score.backward()

    # Collect relevance per time
    R_list = []
    for ht in H:
        if ht.grad is None:
            R_list.append(torch.zeros_like(ht))
        else:
            R_list.append((ht * ht.grad).detach())
    R = torch.stack(R_list, dim=0).detach()

    return R, H.detach()
