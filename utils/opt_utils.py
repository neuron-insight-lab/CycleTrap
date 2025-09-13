import os
import random
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import imageio
import decord
import time
import pynvml
import torch.nn.functional as F

# lpips model download location
# os.environ['TORCH_HOME'] = "./weight"
import lpips



MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
DATASETS_PATH = {
    'coco': 'COCO/path',
    'imagenet': 'ImageNet/path',
    'tgif': 'TGIF/path',
    'msvd': 'MSVD/path',
}

def get_datasets(name, length=200):
    path = DATASETS_PATH[name]
    datasets = []
    is_video = None

    if name == 'coco':
        is_video = False
        filenames = os.listdir(path)
        for p in filenames:
            if not p.endswith('.jpg'):
                continue
            datasets.append(os.path.join(path, p))

    elif name == 'imagenet':
        is_video = False
        filenames = os.listdir(path)
        for p in filenames:
            if not p.endswith('.JPEG'):
                continue
            datasets.append(os.path.join(path, p))
    
    elif name == 'tgif':
        is_video = True
        filenames = os.listdir(path)
        for p in filenames:
            if not p.endswith('.mp4') or not p.endswith('.gif'):
                continue
            datasets.append(os.path.join(path, p))

    elif name == 'msvd':
        is_video = True
        filenames = os.listdir(path)
        for p in filenames:
            if not p.endswith('.avi'):
                continue
            datasets.append(os.path.join(path, p))

    return random.sample(datasets, k=min(length, len(datasets))), is_video


def save_image(image_tensor, f_name):
    mean = torch.tensor(MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=image_tensor.device).view(1, 3, 1, 1)

    image_denorm = image_tensor * std + mean

    image_array = image_denorm[0].permute(1, 2, 0).mul(255).to(torch.uint8).detach().cpu().numpy()
    image = Image.fromarray(image_array)
    image.save(f_name)


def save_video_clip(video_tensor, f_name, max_save_nframe=4):
    nframe = video_tensor.size(0)
    frame_ids = sorted(random.sample(range(nframe), k=min(max_save_nframe, nframe)))

    i = 1
    for ids in frame_ids:
        save_image(video_tensor[ids:ids+1], f'{f_name}_frame{i}')
        i += 1


def is_low_entropy(probs, threshold=0.1):
    """Calculate the average entropy of the sequence"""
    # 只计算最后重复的部分
    if probs.size(1) > 100:
        probs = probs[:, -100:, :]

    entropy = - (probs * probs.log()).sum(-1).mean()
    return entropy.item() < threshold


def resize_tensor(tensors, size):
    if isinstance(size, int):
        size = (size, size)

    tensors = transforms.functional.resize(
        tensors,
        size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return tensors


def clamp(delta, clean_imgs, epsilon):

    clamp_imgs = clamp_add(delta, clean_imgs)
    clamp_delta = torch.clamp(clamp_imgs - clean_imgs, min=-epsilon, max=epsilon)

    return clamp_delta


def clamp_add(tensor_x, tensor_y):
    mean = torch.tensor(MEAN, device=tensor_x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor_x.device).view(1, 3, 1, 1)

    clamp_imgs = ((tensor_x + tensor_y) * std + mean).clamp(0., 1.)
    clamp_imgs = (clamp_imgs - mean) / std

    return clamp_imgs


def image_transform(image, size=None):
    if size is not None and isinstance(size, int):
        size = (size, size)

    transform_list = []
    if size is not None:
        transform_list.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))

    transform = transforms.Compose(transform_list)
    return transform(image)


def get_special_ids(tokenizer, segment_len, not_allowed_tokens=None):
    temp_len = segment_len * 2
    target_ids = None

    # Randomly select characters
    total_vocab_size = tokenizer.vocab_size
    x = torch.ones(total_vocab_size)
    if not_allowed_tokens is not None:
        x[not_allowed_tokens] = 0
    indexs = x.nonzero().squeeze().tolist()
    while True:
        target_ids = random.sample(indexs, k=segment_len)
        adv_str = tokenizer.decode((target_ids * temp_len)[:temp_len])
        if len(tokenizer.encode(adv_str, add_special_tokens=False)) == temp_len:
            break

    return target_ids


def get_nonascii_toks(tokenizer):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    nonascii_toks.extend(tokenizer.all_special_ids)
    
    # record blank
    token = '* '
    s = set()
    while True:
        t = tokenizer(token, add_special_tokens=False).input_ids
        t = t[-1]
        if t not in s:
            s.add(t)
            # print(tokenizer.decode([t]), t)
            nonascii_toks.append(t)
        else:
            break
        token += ' '
 
    return [t for t in nonascii_toks if t < tokenizer.vocab_size]


def pad_tensor_center(tensor, target_h, target_w, is_normalize=True):
    _, c, h, w = tensor.shape

    # Calculate up, down, left and right padding size
    paste_x, r_x = divmod(target_w - w, 2)
    paste_y, r_y = divmod(target_h - h, 2)

    pad_left, pad_right = paste_x, paste_x + r_x
    pad_top, pad_bottom = paste_y, paste_y + r_y

    # first pad 0
    padded = F.pad(
        tensor, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode="constant", value=0
    )

    if is_normalize:
        # pad = (0 - mean) / std
        assert c == len(MEAN)
        pad_values = torch.tensor([-m / s for m, s in zip(MEAN, STD)], 
                                dtype=tensor.dtype, device=tensor.device)
        # Replace the 0 pad with pad_values
        for i in range(c):
            if pad_top > 0:
                padded[:, i, :pad_top, :] = pad_values[i]
            if pad_bottom > 0:
                padded[:, i, -pad_bottom:, :] = pad_values[i]
            if pad_left > 0:
                padded[:, i, :, :pad_left] = pad_values[i]
            if pad_right > 0:
                padded[:, i, :, -pad_right:] = pad_values[i]

    return padded


@torch.no_grad()
def measure_energy_latency(visual_manager, messages, vision_tensors, iter_num=3):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(visual_manager.device.index)
    t1 = time.time()
    gpu_energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    max_length = 0
    max_output_ids= None
    multi_caption = []

    for _ in range(iter_num):
        output_ids, output_str, output_len = visual_manager.get_output_text(messages, vision_tensors)
        multi_caption.append(output_str)
        if output_len > max_length:
            max_length = output_len
            max_output_ids = output_ids
    t2 = time.time()
    latency = (t2 - t1) / iter_num      
    gpu_energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    energy = (gpu_energy_end - gpu_energy_start) / (10 ** 3) / iter_num
    pynvml.nvmlShutdown()
    return latency, energy, max_output_ids, multi_caption, max_length


def cal_lpips(ori_tensors, adv_tensors):

    mean = torch.tensor(MEAN, device=ori_tensors.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=ori_tensors.device).view(1, 3, 1, 1)
    ori_tensors = ori_tensors * std + mean
    adv_tensors = adv_tensors * std + mean

    res_lpips = []
    loss_fn = lpips.LPIPS(net='alex').to(ori_tensors.device)  # 'alex', 'vgg', 'squeeze'
    for i in range(ori_tensors.size(0)):
        a = ori_tensors[i:i+1]
        b = adv_tensors[i:i+1]
        l = loss_fn(a, b)
        res_lpips.append(l.item())

    return sum(res_lpips)/ len(res_lpips)
