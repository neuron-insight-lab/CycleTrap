import torch
import time
import os
from transformers import set_seed
import argparse
import json
import gc
import logging
import random
import torch.nn.functional as F
import nltk
import math
import numpy as np
from collections import defaultdict
import string

import sys
sys.path.append('.')
from utils import *


# 确保下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class StatisticalWeightPool:
    """
    Build a statistical weight pool for the probability of POS tags to EOS
    """
    def __init__(self, pos_to_eos_dir):
        self.pos_to_eos_prob = {}
        self.default_weight = None
        self.pos_to_eos_dir = pos_to_eos_dir

    def _detect_scheme(self, tokens):
        if any(tok.startswith('##') for tok in tokens):    # BERT WordPiece
            return 'wordpiece'
        if any(tok.startswith('Ġ') for tok in tokens):     # GPT2/Byte-BPE
            return 'gpt2'
        if any(tok.startswith('▁') for tok in tokens):     # SentencePiece/LLaMA/Qwen
            return 'spm'
        return 'other'

    def _is_start_of_word(self, tok, scheme):
        if tok in string.punctuation:
            return True
        if scheme == 'wordpiece':
            return not tok.startswith('##')
        if scheme == 'gpt2':
            return tok.startswith('Ġ') or tok in ['Ċ', 'Ġ.']
        if scheme == 'spm':
            return tok.startswith('▁')
        return True

    def _strip_marker(self, tok, scheme):
        if scheme == 'wordpiece':
            return tok[2:] if tok.startswith('##') else tok
        if scheme == 'gpt2':
            return tok.lstrip('Ġ')
        if scheme == 'spm':
            return tok.lstrip('▁')
        return tok

    def tokens_to_words_and_map(self, tokens):
        scheme = self._detect_scheme(tokens)
        words, tok2word = [], []
        word_idx = -1
        for tok in tokens:
            start = self._is_start_of_word(tok, scheme)
            piece = self._strip_marker(tok, scheme)
            if start or word_idx < 0:
                words.append(piece)
                word_idx += 1
            else:
                words[word_idx] += piece
            tok2word.append(word_idx)
        return words, tok2word

    def pos_by_token_from_ids(self, generated_ids, tokenizer):
        
        # 1) Subword sequence
        all_tokens = tokenizer.convert_ids_to_tokens(generated_ids, skip_special_tokens=True)

        # 2) Merge them into words and establish mappings
        words, tok2word = self.tokens_to_words_and_map(all_tokens)

        # 3) get POS
        word_pos = nltk.pos_tag(words)
        word_pos_only = [p for (_, p) in word_pos]

        # 4) Expand the couplet
        pos_by_token = [word_pos_only[widx] for widx in tok2word]

        return pos_by_token

    def load_dataset(self, num=5000):
        coco_path = 'COCO/Path'
        imagenet_path = 'ImageNet/Path'
        
        coco_images = []
        for p in os.listdir(coco_path):
            if not p.endswith('.jpg'):
                continue
            coco_images.append(os.path.join(coco_path, p))

        imagenet_images = []
        for p in os.listdir(imagenet_path):
            if not p.endswith('.JPEG'):
                continue
            imagenet_images.append(os.path.join(imagenet_path, p))
        
        return random.sample(coco_images, min(num, len(coco_images))) + random.sample(imagenet_images, min(num, len(imagenet_images)))
    
    def _normalize(self):
        self.pos_to_eos_prob['default_weight'] = np.median(list(self.pos_to_eos_prob.values()))

        total_values = sum(list(self.pos_to_eos_prob.values()))
        self.pos_to_eos_prob = {k: v/total_values for k,v in self.pos_to_eos_prob.items()}
        self.default_weight = self.pos_to_eos_prob['default_weight']

        
    def build_from_data(self, visual_manager, model_name):
        """
        Build a POS-EOS statistical model from the dataset
        """
        file_name = os.path.join(self.pos_to_eos_dir, f'{model_name}_pos2eos.json')
        if os.path.isfile(file_name):
            with open(file_name) as f:
                self.pos_to_eos_prob = json.load(f)
            logger.info(f"Load weight pool from {file_name} file")
            self._normalize()
            return

        pos_eos_stats = defaultdict(list)

        eos_token_id = visual_manager.eos_token_id

        dataset_images = self.load_dataset(num=5000)
        
        logger.info("Building Statistical Weight Pool...")
        for i, image_path in enumerate(dataset_images):
            if i % 100 == 0:
                logger.info(f"Processing image {i}/{len(dataset_images)}")

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What is the content of this image?"}
                ]}
            ]
            vision_tensors = visual_manager.get_vision_tensors(messages)
            output_ids, output_str, output_len = visual_manager.get_output_text(messages, vision_tensors)
        
            with torch.no_grad():
                inputs = visual_manager.get_model_inputs(messages, vision_tensors)

                inputs = visual_manager.process_inputs(inputs, output_ids)
                
                logits = visual_manager.model(**inputs).logits.squeeze()
                logits = logits[-1-output_len:-1, :]

                generated_ids = output_ids[0, -output_len:]

                pos_tags = self.pos_by_token_from_ids(generated_ids, visual_manager.processor.tokenizer)
                
                eos_probs = torch.softmax(logits, dim=-1)[:, eos_token_id]
                if isinstance(eos_token_id, list):
                    eos_probs = eos_probs.sum(-1)
                
                # Calculate the EOS probability after each POS tag
                for j, pos_tag in enumerate(pos_tags):  # 除最后一个token
                    if j + 1 < len(eos_probs):
                        next_eos_prob = eos_probs[j + 1].item()
                        pos_eos_stats[pos_tag].append(next_eos_prob)
            if (i+1) % 100 == 0 or i == len(dataset_images)-1:
                for pos_tag, eos_probs in pos_eos_stats.items():
                    self.pos_to_eos_prob[pos_tag] = np.mean(eos_probs)
                
                logger.info(f"Built weight pool with {len(self.pos_to_eos_prob)} POS tags")
                with open(file_name, 'w') as f:
                    json.dump(self.pos_to_eos_prob, f, indent=4, ensure_ascii=False)
        
        self._normalize()
    
    def get_weight(self, pos_tag):
        """
        Obtain the weight based on the POS label
        """
        weight = self.pos_to_eos_prob.get(pos_tag, self.default_weight)
        
        return weight
    

def get_loss(tokenizer, outputs, weight_pool, output_ids, eos_token_id, theta_w=1e5, lambda_rep=1):
    loss_lps, loss_rep = None, None

    # =============Loss component 1: LPS=============
    output_len = output_ids.shape[0]
    logits = outputs.logits.squeeze()[-1-output_len:-1 ,:]
    eos_probs = torch.softmax(logits, dim=-1)[:, eos_token_id]
    if isinstance(eos_token_id, list):
        eos_probs = eos_probs.sum(dim=-1)
    
    # Convert to text and perform POS annotation
    pos_tags = weight_pool.pos_by_token_from_ids(output_ids, tokenizer)
    
    # Calculate the weighted EOS loss
    weighted_eos_loss = 0.0
    total_weight = 0.0
    
    for i in range(len(eos_probs)-1):
        if i > 0 and i - 1 < len(pos_tags):
            prev_pos = pos_tags[i - 1]
            weight = weight_pool.get_weight(prev_pos)
        else:
            weight = weight_pool.default_weight
        
        weighted_eos_loss += weight * eos_probs[i]
        total_weight += weight
    
    loss_lps = weighted_eos_loss / total_weight

    # =============Loss component 2: REP=============
    if 'InstructBlip' in str(type(outputs)):
        hidden_states = outputs.language_model_outputs.hidden_states
    else:
        hidden_states = outputs.hidden_states
    output_hidden_states = torch.stack(hidden_states)[:, 0, -output_len:, :]
    
    norms = torch.norm(output_hidden_states, p=2, dim=-1)
    loss_rep = lambda_rep * norms.mean()

    return loss_lps, loss_rep


def attack(visual_manager, messages, weight_pool, args, epoch):

    # ============ PGD parameter ============
    eps = args.epsilon / 255.         # maximal perturbation magnitude
    alpha = args.alpha / 255.        # The step size of each iteration
    steps = args.steps           # maximum iterations
    mu = args.m
    eval_interval = args.eval_interval
    alpha_weight = args.alpha_weight

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

        loss_lps, loss_rep = get_loss(visual_manager.processor.tokenizer, outputs, weight_pool, output_ids[0, -output_len:], visual_manager.eos_token_id)
        
        loss1_val, loss2_val = loss_lps.detach().clone(), loss_rep.detach().clone()
        
        # Time decay function
        ratio1 = 10.0 * math.log(step + 1) - 20.0
        if step == 0:
            lambda_t = torch.abs(loss1_val / loss2_val / ratio1)
        else:
            cur_lambda_t = torch.abs(loss1_val / loss2_val / ratio1)
            lambda_t = 0.9 * last_lambda_t + 0.1 * cur_lambda_t
        
        last_lambda_t = lambda_t

        # total loss
        loss = alpha_weight * loss_lps + lambda_t * loss_rep
        
        visual_manager.model.zero_grad()
        loss.backward(retain_graph=False)

        # Update the momentum term
        grad = delta.grad.detach()
        momentum = mu * momentum + grad / (grad.abs().mean() + 1e-8)

        # PGD update
        with torch.no_grad():
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
            logger.info(f"Step {step+1} | Loss: {loss.item():.5f} | Speed: {(step+1)/dur:.2f} it/s")
            if args.max_new_tokens == out_len:
                is_success = True
                break
            # updated output parameter
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

    # 1. Build a statistical weight pool
    weight_pool = StatisticalWeightPool(args.pos_to_eos_dir)
    weight_pool.build_from_data(visual_manager, args.model_name)

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

        attack(visual_manager, messages, weight_pool, args, epoch)

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
    parser.add_argument("--root_dir", type=str, default='save/baseline/LingoLoop/')
    parser.add_argument("--seed", type=int, default=2025, help='random seed')
    parser.add_argument("--device_id", type=int, default=1, help='The device id being measured')
    
    # =========== new parameters =================
    parser.add_argument("--alpha_weight", type=float, default=1e5)
    parser.add_argument("--pos_to_eos_dir", type=str, default='POS/')

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
        os.makedirs(os.path.join(save_dir, 'res'))
    args.save_dir = save_dir

    if not os.path.exists(args.pos_to_eos_dir):
        os.makedirs(args.pos_to_eos_dir)

    main(args)