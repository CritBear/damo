import torch


def find_bind_mgp(
        bind_j3gp, jgr, jgp, mgp, j3_indices, j3_weights, init_params=None,
        eps=1e-5, max_iter=10, mse_threshold=1e-4, u=1e-3, v=1.5
):
    n_markers, _ = mgp.shape
    if init_params is not None:
        params = init_params.flatten().clone()
    else:
        params = torch.zeros(n_markers * 3).to(mgp.device)

    out_n = n_markers * 3
    jacobian = torch.zeros([out_n, params.shape[0]]).to(mgp.device)

    last_update = 0
    last_mse = 0

    for i in range(max_iter):
        # params = torch.clamp(params, min=-2, max=2)

        residual, mse = get_residual(bind_j3gp, jgr, jgp, params, mgp, j3_indices, j3_weights, mse=True)

        if abs(mse - last_mse) < mse_threshold:
            return params, lbs(bind_j3gp, jgr, jgp, params, j3_indices, j3_weights), i, mse

        for k in range(params.shape[0]):
            jacobian[:, k] = get_derivative(k, bind_j3gp, jgr, jgp, params, mgp, j3_indices, j3_weights, eps)

        jtj = torch.matmul(jacobian.T, jacobian)
        jtj = jtj + u * torch.eye(jtj.shape[0]).to(mgp.device)

        update = last_mse - mse
        delta = torch.matmul(
            torch.matmul(torch.linalg.inv(jtj), jacobian.T), residual
        ).ravel()
        params -= delta

        if update > last_update and update > 0:
            u /= v
        else:
            u *= v

        last_update = update
        last_mse = mse

    return params, lbs(bind_j3gp, jgr, jgp, params, j3_indices, j3_weights), max_iter, last_mse


def get_residual(bind_j3gp, jgr, jgp, params, mgp, j3_indices, j3_weights, mse=False):
    virtual_mgp = lbs(bind_j3gp, jgr, jgp, params, j3_indices, j3_weights)

    residual = virtual_mgp - mgp

    if mse:
        residual_mse = torch.mean(torch.linalg.norm(residual, dim=-1)).item()
        residual = residual.view(-1, 1)
        return residual, residual_mse
    else:
        residual = residual.view(-1, 1)
        return residual


def get_derivative(k, bind_j3gp, jgr, jgp, params, mgp, j3_indices, j3_weights, eps):
    params1 = params.clone()
    params2 = params.clone()

    params1[k] += eps
    params2[k] -= eps

    res1 = get_residual(bind_j3gp, jgr, jgp, params1, mgp, j3_indices, j3_weights)
    res2 = get_residual(bind_j3gp, jgr, jgp, params2, mgp, j3_indices, j3_weights)

    d = (res1 - res2) / (2 * eps)

    return d.ravel()


def lbs(bind_j3gp, jgr, jgp, params, j3_indices, j3_weights):
    j3gr = jgr[j3_indices, :, :]  # [m, j3, 3, 3]
    j3gp = jgp[j3_indices, :]  # [m, j3, 3]

    local_offsets = params.view(-1, 3)[:, None, :] - bind_j3gp

    points = j3gr @ local_offsets[:, :, :, None]  # [m, j3, 3, 1]
    points = points.squeeze()
    points += j3gp  # [m, j3, 3]
    points = torch.sum(points * j3_weights[:, :, None], dim=1)  # [m, 3]

    return points

