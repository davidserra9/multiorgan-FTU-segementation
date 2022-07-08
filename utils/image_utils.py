import json

def read_mask(path_mask):
    """
    Read a mask (.json) from a path.
    :param path_mask: path to the mask
    :return: mask
    """
    with open(path_mask) as f:
        mask = json.load(f)
    return mask


def mask_to_rle(mask):
    """
    Convert a binary mask to a rle.
    :param mask: binary mask
    :return: rle
    """
    rle = []

    # todo: Convert a binary mask to a rle.

    return rle