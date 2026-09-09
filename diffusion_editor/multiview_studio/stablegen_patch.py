"""Image-space Patch handling shared by the Studio UI and isolated worker."""
import math
import numpy as np
from PIL import Image


def patch_rect(bounds, size):
    width, height = size
    if bounds is None:
        return (0, 0, width, height)
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError('Invalid Patch rectangle')
    x0, y0, x1, y1 = bounds
    rect = (max(0, math.floor(min(x0,x1))), max(0, math.floor(min(y0,y1))),
            min(width, math.ceil(max(x0,x1))), min(height, math.ceil(max(y0,y1))))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        raise ValueError('Patch is empty or outside the image')
    return rect


def prepare_patch(image, mask, depth, bounds, resolution):
    if image.size != mask.size or image.size != depth.size:
        raise ValueError('RGB, mask and depth must have identical dimensions')
    rect = patch_rect(bounds, image.size)
    width, height = rect[2]-rect[0], rect[3]-rect[1]
    scale = resolution / max(width, height)
    size = (max(64, round(width*scale/64)*64), max(64, round(height*scale/64)*64))
    rgb = image.crop(rect).resize(size, Image.Resampling.LANCZOS)
    cropped_mask = mask.crop(rect).resize(size, Image.Resampling.BILINEAR)
    control = depth.crop(rect).resize(size, Image.Resampling.BILINEAR)
    if not np.asarray(cropped_mask).any():
        raise ValueError('The mask does not intersect the Patch')
    return rgb, cropped_mask, control, rect


def paste_patch(source, result, mask, rect):
    result = result.resize((rect[2]-rect[0],rect[3]-rect[1]),Image.Resampling.LANCZOS)
    crop = np.array(result.convert('RGB'))
    keep = np.asarray(mask.crop(rect)) == 0
    crop[keep] = np.asarray(source.convert('RGB').crop(rect))[keep]
    combined = source.convert('RGB').copy()
    combined.paste(Image.fromarray(crop),rect[:2])
    return combined
