import torch
from torch.nn import functional as F
import time
import os
from transformers import set_seed
import argparse
import json
import gc
import logging

from utils import *


def get_loss(logits, target_ids):
    # ==========cycle loss==============
    probs = F.softmax(logits, dim=-1)
    target_probs = probs[:, :, target_ids].sum(-1)
    loss = torch.nn.BCELoss(reduction='none')(target_probs, torch.ones_like(target_probs))    # -torch.log(special_p)).mean(dim=-1)
    loss = loss.mean()

    return loss


def attack(visual_manager, messages, target_ids, args, epoch):

    # ============ 攻击参数 ============
    eps = args.epsilon / 255.         # 最大扰动范围
    alpha = args.alpha / 255.        # 每次迭代步长
    steps = args.steps           # PGD迭代次数
    eval_interval = args.eval_interval
    mu = args.mu

    is_success = False

    # ========== 预处理 ================
    vision_tensors = visual_manager.get_vision_tensors(messages)
    latency, energy, output_ids, output_str, output_len = measure_energy_latency(visual_manager, messages, vision_tensors, iter_num=args.sample_times)

    if visual_manager.is_video:
        vision_path = messages[0]['content'][0]['video']
    else:
        vision_path = messages[0]['content'][0]['image']
   
    res = dict()
    res[0] = {
        'target_tokens': "  ".join([visual_manager.processor.decode(idx) for idx in target_ids]),
        'target_ids': target_ids,
        'prompt': messages[0]['content'][1]['text'],
        'vision_path': vision_path,
        'output_str': output_str,
        'output_len': output_len,
        'latency': latency,
        'energy': energy,
    }

    delta = torch.zeros_like(vision_tensors, requires_grad=True)
    # delta = torch.randn_like(vision_tensors, requires_grad=True)

    momentum = torch.zeros_like(delta, requires_grad=False)
    start_time = time.time()

    for step in range(steps):

        inputs = visual_manager.get_model_inputs(messages, vision_tensors + delta)
        inputs = visual_manager.process_inputs(inputs, output_ids)
        
        logits = visual_manager.model(**inputs).logits
        output_logits = logits[:, -1-output_len:-1, :]

        loss = get_loss(output_logits, target_ids)

        # 反向传播
        visual_manager.model.zero_grad()
        loss.backward(retain_graph=False)

        # 更新动量项
        grad = delta.grad.detach()
        momentum = mu * momentum + grad / (grad.abs().mean() + 1e-8)

        # PGD更新
        delta.data = delta - alpha * momentum.sign()
        delta.data = clamp(delta.data, vision_tensors.data, epsilon=eps)
        delta.grad.zero_()

        if (step + 1) % eval_interval == 0:
            # 测试
            latency, energy, out_ids, output_str, out_len = measure_energy_latency(visual_manager,
                             messages, vision_tensors + delta, iter_num=args.sample_times)
            dur= time.time() - start_time
            res[step+1] = {
                'output_str': output_str,
                'output_len': out_len,
                'latency': latency,
                'energy': energy,
                'loss': loss.item(),
                'time': dur,
            }
            logger.info(f"Step {step+1} | Loss: {loss.item():.2f} | Speed: {(step+1)/dur:.2f} it/s")
            if args.max_new_tokens == out_len:
                is_success = True
                break
            # 更新参数 (可选)
            output_ids = out_ids
            output_len = out_len


    res[0]['is_success'] = is_success
    during = time.time() - start_time
    logger.info(f'**is_success:{is_success}\ttotal time: {int(during // 60)}m {(int(during % 60))}s')

    # =========== 计算 lpips ===========
    # lpips_val = cal_lpips(vision_tensors, delta+vision_tensors)
    # res[0]['lpips'] = lpips_val

    # =========== 保存图片 =============
    file_suffix = vision_path.split('.')[-1]
    if visual_manager.is_video:
        save_image(delta, f"{args.save_dir}/vis/{epoch}_delta.jpg")
        save_video_clip(delta+vision_tensors, f"{args.save_dir}/vis/{epoch}_adv_video.{file_suffix}")
    else:
        save_image(delta, f"{args.save_dir}/vis/{epoch}_delta.{file_suffix}")
        save_image(delta+vision_tensors, f"{args.save_dir}/vis/{epoch}_adv_image.{file_suffix}")
    
    # =========== 保存结果 ===========
    with open(f"{args.save_dir}/res/{epoch}_res.json", 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


def main(args):

    # 获取数据集
    datasets, is_video = get_datasets(args.data_name, length=args.data_len)

    # 记录之前的结果（如果有的话）
    start_epoch = 0
    res_dir = os.path.join(args.save_dir, 'res')
    is_before = len(os.listdir(res_dir)) > 0
    if is_before:
        path = sorted(os.listdir(res_dir), key=lambda x: int(x.split('_')[0]))[-1]
        start_epoch = int(path.split('_')[0]) + 1
        if start_epoch >= len(datasets):
            exit(0)

    # 获取 visual manager (model and processor)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    visual_manager = get_visual_manager(args.model_name, device, is_video)    
    visual_manager.init_generation_config(args.max_new_tokens, args.repetition_penalty, args.temperature)

    for epoch in range(start_epoch, len(datasets)):
        
        logger.info(f'\n===============epoch {epoch} / {len(datasets)}==================')
        vision_path = datasets[epoch]
        logger.info(vision_path)

        # ============ 准备生成模版 ============
        # prompt = ['What is the content of this image?', 'Describe this image.', 'Please provide a description for this image.', 'Describe the content of this image.']
        messages = [
            {"role": "user", "content": []}
        ]
        if is_video:
            messages[0]['content'].append({"type": "video", "video": vision_path})
            messages[0]['content'].append({"type": "text", "text": f"Describe this video."})
        else:
            messages[0]['content'].append({"type": "image", "image": vision_path})
            messages[0]['content'].append({"type": "text", "text": f"Describe this image."})

        # 选取重复种子
        target_ids = get_special_ids(visual_manager, messages, args.segment_len, is_random=False)
        # 执行循环诱导优化
        attack(visual_manager, messages, target_ids, args, epoch)

        gc.collect()
        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    # InternVL3-8B  Qwen2.5-VL-7B  LLaVa-Next  LLaVa-Next-Video  InstructBLIP  InstructBlipVideo 
    parser.add_argument('--model_name', default='Qwen2.5-VL-7B', 
                        choices=MODEL_PATHS.keys())
    parser.add_argument("--data_name", type=str, default="coco",
                        choices=['coco', 'imagenet', 'tgif', 'msvd'])
    parser.add_argument("--data_len", type=int, default=200)

    parser.add_argument('--max_new_tokens', default=1024, type=int, help='The maximum number of tokens that the model can produce')
    parser.add_argument('--segment_len', default=5, type=int, help='The number of seed token')
    parser.add_argument("--is_ascii", type=bool, default=True, help='Determine whether the selected token is an ASCII code')
    parser.add_argument("--eval_interval", default=10, type=int, help='How many times should be evaluated at intervals')

    # The parameters for PGD optimization
    parser.add_argument('--steps', default=300, type=int)
    parser.add_argument('--epsilon', default=16, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--mu', default=0.9, type=float, help='Momentum weight parameter')

    parser.add_argument("--sample_times", type=int, default=1, help='The number of inference during the evaluation period')
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help='The repetition penalty parameter of the generation process')
    parser.add_argument("--root_dir", type=str, default='save/CycleTrap/', help='Result saving directory')
    parser.add_argument("--seed", type=int, default=2025, help='random seed')
    parser.add_argument("--device_id", type=int, default=1, help='The device id being measured')
    parser.add_argument("--temperature", type=float, default=None, help='temperature')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # print(args)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            # logging.FileHandler('default'),
            logging.StreamHandler()
        ])
    logger.info(str(args))

    set_seed(args.seed)
    save_dir = os.path.join(args.root_dir, str(args.model_name) + '_' + str(args.data_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'vis'))
        os.makedirs(os.path.join(save_dir, 'res'))
    args.save_dir = save_dir

    main(args)