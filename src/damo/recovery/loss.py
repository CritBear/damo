def configuration_loss(prediction, batch, mode='legacy'):
    if mode != 'legacy':
        raise ValueError('This release supports only the baseline legacy configuration loss')
    indices, weights, offsets = prediction
    mask = batch['points_mask'][:, batch['points_mask'].shape[1] // 2].bool()
    count = mask.sum().clamp_min(1)
    target_w = batch['m_j3_weights']
    li = (indices - batch['m_j_weights']).square().masked_fill(~mask[..., None], 0).sum() / count
    lw = (weights - target_w).square().masked_fill(~mask[..., None], 0).sum() / count
    do = (offsets - batch['m_j3_offsets']).square().masked_fill(~mask[..., None, None], 0)
    lo = (do * target_w[..., None]).sum() / count
    return dict(total=li + lw + lo, indices=li, weights=lw, offsets=lo)
