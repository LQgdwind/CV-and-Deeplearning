import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# 改变亮暗、对比度和颜色等
def random_distort(img,lower=0.5,upper=1.5,brightness=True,contrast=True,color=True):

    # 改变亮度
    def random_brightness(img, lower=lower, upper=upper):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 改变对比度
    def random_contrast(img, lower=lower, upper=upper):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 改变颜色
    def random_color(img, lower=lower, upper=upper):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)
    ops = []
    if brightness == True:
        ops.append(random_brightness)
    if contrast == True:
        ops.append(random_contrast)
    if color == True:
        ops.append(random_color)
    np.random.shuffle(ops)
    img = Image.fromarray(img)

    for i,item in enumerate(ops):
        img = ops[i](img)
    img = np.asarray(img)
    return img


# 填充
def random_expand(img,max_ratio=4.,fill=None,keep_ratio=True,thresh=0.5):
    if random.random() > thresh:
        return img
    if max_ratio < 1.0:
        return img
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)
    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0
    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    return out_img.astype('uint8')


# 裁剪
def random_crop(img,scales=[0.3, 1.0],max_ratio=2.0,max_trial=50):

    img = Image.fromarray(img)
    # print(img)
    w, h = img.size
    crops = [(0, 0, w, h)]
    for _ in range(max_trial):
        scale = random.uniform(scales[0], scales[1])
        aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale),min(max_ratio, 1 / scale / scale))
        crop_h = int(h * scale / np.sqrt(aspect_ratio))
        crop_w = int(w * scale * np.sqrt(aspect_ratio))
        crop_x = random.randrange(w - crop_w)
        crop_y = random.randrange(h - crop_h)
        crops.append((crop_x, crop_y, crop_w, crop_h))
    # print(crops)
    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                        crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img


# 缩放
def random_interp(img, size, interp=None):
    # 缩放算法列表
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    # 如果没有指定缩放算法，则选一种
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img


# 翻转
def random_flip(img, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
    return img

# 图像增广
def image_augment(img, size=512, means=None):
    # 缩放
    img = random_interp(img, size)
    # 改变亮暗、对比度和颜色
    img = random_distort(img)
    # 裁剪
    img = random_crop(img)
    # 填充
    img = random_expand(img,fill=means)
    # 缩放
    img = random_interp(img, size)
    # 翻转
    img = random_flip(img)
    # mean和std由Caculating_mean_std文件计算得出
    mean = [0.5420377, 0.5206433, 0.49089015]
    std = [0.21957076, 0.21739748, 0.2220452]
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    # print(mean.shape)
    img = (img / 255.0 - mean) / std
    # 相当于按元素取绝对值，防止其为负数
    img = np.maximum(img,-img)
    img = img.astype('float32')
    # print(img.shape)
    return img

# 图像轻度增广
def image_light_augment(img, size=512, means=None):
    # 改变亮暗、对比度
    img = random_distort(img,brightness=True,contrast=True,color=False,lower=0.9,upper=1.1)
    # 缩放
    img = random_interp(img, size)
    # 翻转
    img = random_flip(img)
    return img