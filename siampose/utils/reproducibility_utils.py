import pytorch_lightning


def set_seed(
    seed,
    set_deterministic=False,  # False = optimal performance
    set_benchmark=True,  # True = optimal performance, unless input tensor shapes vary
):  # pragma: no cover
    """Set the provided seed in python/numpy/DL framework.

    :param seed: (int) the seed
    """
    pytorch_lightning.seed_everything(seed)
    # the four setters below are included in 'seed_everything' above
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    import torch.backends.cudnn

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
