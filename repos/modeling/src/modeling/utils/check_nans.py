import torch


def check_nans(tensor: torch.Tensor, name: str):
    """
    Check if the tensor contains NaN values and log an error if it does.
    """
    if not torch.isfinite(tensor).all():
        finite_elements = torch.isfinite(tensor)
        num_finite = finite_elements.sum()
        total_elements = tensor.numel()
        # logger.error(
        #     f"{name} contains NaN values: number of finite: {num_finite} number of total: {total_elements}"
        # )
        # return False
        raise ValueError(
            f"{name} contains NaN values: number of finite: {num_finite} number of total: {total_elements}"
        )
    return True
