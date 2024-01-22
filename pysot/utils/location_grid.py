# import torch
import paddle
def compute_locations(features,stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device):
    shifts_x = paddle.arange(
        0, w * stride, step=stride,
        dtype=paddle.float32, device=device
    )
    shifts_y = paddle.arange(
        0, h * stride, step=stride,
        dtype=paddle.float32, device=device
    )
    shift_y, shift_x = paddle.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride

    # locations = torch.stack((shift_x, shift_y), dim=1) + 32  #alex:48 // 32  # yuan
    locations = paddle.stack([shift_x, shift_y], axis=1) + 32  #alex:48 // 32  # yuan
    return locations



