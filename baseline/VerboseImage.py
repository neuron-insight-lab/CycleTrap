import torch
from torch.nn import functional as F
from tqdm import tqdm
import time
import os
from transformers import set_seed
import argparse
import json
import gc
import logging
import math

import sys
sys.path.append('.')
from utils import *


def get_loss(outputs, output_len, eos_token_id):
    logits = outputs.logits.squeeze()[-1-output_len:-1 ,:]

    t_logits = logits + 1e-9
    uni_distribution = (torch.ones(t_logits.shape) / t_logits.shape[1]).to(t_logits.device)
    loss_uncertainty = F.kl_div(t_logits.log_softmax(dim=-1), uni_distribution, reduction='sum')
    
    logits = torch.softmax(logits, dim=-1)
    loss_eos = logits[:, eos_token_id]
    if isinstance(eos_token_id, list):
       loss_eos = loss_eos.sum(dim=-1) 
    loss_eos = loss_eos.mean()
    
    if 'InstructBlipForConditionalGeneration' in str(type(outputs)):
        hidden_states = outputs.language_model_outputs.hidden_states
    else:
        hidden_states = outputs.hidden_states

    hidden_states = torch.stack(hidden_states, dim = 0).squeeze().float()   
    hidden_states = torch.abs(hidden_states)[:, -output_len:, :]
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    loss_diversity = -torch.norm(hidden_states, p='nuc')

    return {"loss1": loss_uncertainty, "loss2": loss_eos, "loss3": loss_diversity}


def attack(visual_manager, messages, args, epoch):

    # ============ PGD parameter ============
    eps = args.epsilon / 255.         # maximal perturbation magnitude
    alpha = args.alpha / 255.        # The step size of each iteration
    steps = args.steps           # maximum iterations
    mu = args.m
    eval_interval = args.eval_interval

    is_success = False

    # ==========Preprocessing================
    vision_tensors = visual_manager.get_vision_tensors(messages)
    latency, energy, output_ids, output_str, output_len = measure_energy_latency(visual_manager, messages, vision_tensors, iter_num=args.sample_times)

    # print(output_str)
    if visual_manager.is_video:
        image_path = messages[0]['content'][0]['video']
    else:
        image_path = messages[0]['content'][0]['image']
   
    res = dict()
    res[0] = {
        'image_path': image_path,
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

        outputs = visual_manager.model(**inputs, return_dict=True, output_hidden_states=True)

        result = get_loss(outputs, output_len, visual_manager.eos_token_id)

        loss1, loss2, loss3 = result["loss1"], result["loss2"], result["loss3"]
        loss1_val, loss2_val, loss3_val = loss1.detach().clone(), loss2.detach().clone(), loss3.detach().clone()

        ratio1 = 10.0 * math.log(step + 1) - 20.0
        ratio2 = 0.5 * math.log(step + 1) + 1.0

        if step == 0:
            lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
            lambda2 = torch.abs(loss1_val / loss3_val / ratio2)
        else:
            cur_lambda1 = torch.abs(loss1_val / loss2_val / ratio1)
            cur_lambda2 = torch.abs(loss1_val / loss3_val / ratio2)                     
            lambda1 = 0.9 * last_lambda1 + 0.1 * cur_lambda1
            lambda2 = 0.9 * last_lambda2 + 0.1 * cur_lambda2
        
        last_lambda1, last_lambda2 = lambda1, lambda2  
        
        loss = loss1 + lambda1 * loss2 + lambda2 * loss3

        # Backwards propagation
        visual_manager.model.zero_grad()
        loss.backward(retain_graph=False)

        # Update the momentum term
        grad = delta.grad.detach()
        momentum = mu * momentum + grad / (grad.abs().mean() + 1e-8)

        # PGD update
        delta.data = delta - alpha * momentum.sign()
        delta.data = clamp(delta.data, vision_tensors.data, epsilon=eps)
        delta.grad.zero_()

        if (step + 1) % eval_interval == 0:
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
            # Update output parameters
            output_ids = out_ids
            output_len = out_len


    res[0]['is_success'] = is_success

    during = time.time() - start_time
    logger.info(f'**is_success:{is_success}\ttotal time: {int(during // 60)}m {(int(during % 60))}s')
    with open(f"{args.save_dir}/res/{epoch}_res.json", 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


def main(args):

    # Obtain the dataset
    datasets, is_video = get_datasets(args.data_name, length=args.data_len)

    # Record the previous results if any
    start_epoch = 0
    res_dir = os.path.join(args.save_dir, 'res')
    is_before = len(os.listdir(res_dir)) > 0
    if is_before:
        path = sorted(os.listdir(res_dir), key=lambda x: int(x.split('_')[0]))[-1]
        start_epoch = int(path.split('_')[0]) + 1
        if start_epoch >= len(datasets):
            exit(0)

    # Obtain the visual manager (model and processor)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    visual_manager = get_visual_manager(args.model_name, device, is_video)    
    visual_manager.init_generation_config(args.max_new_tokens, args.repetition_penalty)

    for epoch in range(start_epoch, len(datasets)):
        
        logger.info(f'\n===============epoch {epoch} / {len(datasets)}==================')
        image_path = datasets[epoch]
        logger.info(image_path)

        # ============ Prepare to input the template ============
        messages = [
            {"role": "user", "content": []}
        ]
        if is_video:
            messages[0]['content'].append({"type": "video", "video": image_path})
            messages[0]['content'].append({"type": "text", "text": f"Describe this video."})
        else:
            messages[0]['content'].append({"type": "image", "image": image_path})
            messages[0]['content'].append({"type": "text", "text": f"Describe this image."})

        attack(visual_manager, messages, args, epoch)

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
    parser.add_argument("--is_ascii", type=bool, default=True, help='Determine whether the selected token is an ASCII code')
    parser.add_argument("--eval_interval", default=10, type=int, help='How many times should be evaluated at intervals')

    # The parameters for PGD optimization
    parser.add_argument('--steps', default=300, type=int)
    parser.add_argument('--epsilon', default=16, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--m', default=0.9, type=float, help='Momentum weight parameter')

    parser.add_argument("--sample_times", type=int, default=1, help='The number of inference during the evaluation period')
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help='The repetition penalty parameter of the generation process')
    parser.add_argument("--root_dir", type=str, default='save/baseline/VerboseImage/')
    parser.add_argument("--seed", type=int, default=2025, help='random seed')
    parser.add_argument("--device_id", type=int, default=1, help='The device id being measured')
    
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
            logging.FileHandler('default'),
            logging.StreamHandler()
        ])
    logger.info(str(args))

    set_seed(args.seed)
    save_dir = os.path.join(args.root_dir, str(args.model_name) + '_' + str(args.data_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # os.makedirs(os.path.join(save_dir, 'vis'))
        os.makedirs(os.path.join(save_dir, 'res'))
    args.save_dir = save_dir

    main(args)