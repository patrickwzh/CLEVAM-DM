import torch

from .generators import build_generator_arch


def get_z(heat: float, img_size: tuple, batch: int, device: str):
    def calc_z_shapes(img_size, n_levels):
        h, w = img_size
        z_shapes = []
        channel = 3

        for _ in range(n_levels - 1):
            h //= 2
            w //= 2
            channel *= 2
            z_shapes.append((channel, h, w))
        h //= 2
        w //= 2
        z_shapes.append((channel * 4, h, w))
        return z_shapes

    z_list = []
    z_shapes = calc_z_shapes(img_size, 3)
    for z in z_shapes:
        z_new = torch.randn(batch, *z, device=device) * heat
        z_list.append(z_new)
    return z_list


def interpolate_single_frame(net, img0, img1, fflow, bflow, F1t, F2t, time):
    zs = get_z(0.3, img0.shape[-2:], img0.shape[0], img0.device)
    conds = [img0, img1, fflow, bflow]
    pred, _ = net(zs=zs, inps=conds, F1t=F1t, F2t=F2t, time=time, code="decode")
    return torch.clamp(pred, 0.0, 1.0)
