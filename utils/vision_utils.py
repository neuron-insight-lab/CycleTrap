import torch
from torchvision import transforms
from PIL import Image
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, \
        InstructBlipForConditionalGeneration, InstructBlipProcessor, \
        InstructBlipVideoForConditionalGeneration, InstructBlipVideoProcessor

from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas
from transformers.image_processing_utils import select_best_resolution

import decord

from torchvision.transforms.v2 import functional as F

from .opt_utils import image_transform, resize_tensor, pad_tensor_center


MODEL_PATHS = {
    'Qwen2.5-VL-7B': 'Qwen/Qwen2.5-VL-7B-Instruct', # from huggingface
    'InternVL3-8B': 'OpenGVLab/InternVL3-8B-hf',
    'InstructBLIP': 'Salesforce/instructblip-vicuna-7b',
    'InstructBlipVideo': 'Salesforce/instructblip-vicuna-7b',
    'LLaVa-Next':'llava-hf/llava-v1.6-mistral-7b-hf',
    'LLaVa-Next-Video': '# llava-hf/LLaVA-NeXT-Video-7B-hf'
}


def smart_nframes(total_frames, video_fps):
    """calculate the number of frames for video used for model inputs."""
    FRAME_FACTOR = 2    
    min_frames = 4
    max_frames = 768
    nframes = total_frames / video_fps * FRAME_FACTOR
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round(nframes / FRAME_FACTOR) * FRAME_FACTOR
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def get_video_frames(video_path, nframes=None, max_frames=15, min_frame=4, return_tensors='pt'):
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    if nframes is None:
        nframes = smart_nframes(total_frames, video_fps)
    nframes = max(min_frame, min(max_frames, nframes))
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    if return_tensors == 'pt':
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


class VisualManager:
    def __init__(self, model, processor, is_video):
        self.model = model
        self.processor = processor
        self.is_video = is_video

        self.device = model.device
        self.max_new_tokens = 1024
        self.eos_token_id = model.generation_config.eos_token_id
        self.repetition_penalty = 1.05

    def get_model_inputs(self, messages, vision_tensors):
        raise NotImplementedError("Must implement get_model_inputs() method")
    
    def get_vision_tensors(self, messages):
        raise NotImplementedError("Must implement get_vision_tensors() method")

    def get_output_text(self, messages, vision_tensors, greedy_search=False):
        inputs = self.get_model_inputs(messages, vision_tensors)

        # model generation config
        gen_config = {
            'max_new_tokens': self.max_new_tokens,
            'eos_token_id': self.eos_token_id,
            'repetition_penalty': self.repetition_penalty,
        }
        if greedy_search:
            gen_config.update({
                'do_sample': False,
                'temperature': None,
                'top_k': None,
                'top_p': None
            })
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **gen_config)

        output_len = generated_ids.size(1) - inputs.input_ids.size(1)
        output_str = self.processor.decode(generated_ids[0, inputs.input_ids.size(1):])

        return generated_ids, output_str, output_len
    

    def process_inputs(self, inputs, output_ids):
        inputs.update({'input_ids': output_ids})
        inputs.pop('attention_mask')
        return inputs
    
    
    def init_generation_config(self, max_new_tokens=None, repetition_penalty=None):
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id if self.model.generation_config.pad_token_id is None else self.model.generation_config.pad_token_id
        if max_new_tokens:
            self.max_new_tokens = max_new_tokens
        if repetition_penalty:
            self.repetition_penalty = repetition_penalty




# =========== Qwen2.5-VL ===========
class QwenManager(VisualManager):

    # def __init__(self, model, processor, is_video):
    #     super().__init__(model, processor, is_video)


    def smart_resize(self, height, width, factor=28, min_pixels = 56 * 56, max_pixels = 14 * 14 * 4 * 1280):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        """
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor

        return h_bar, w_bar


    def process_vision(self, vision_tensors, fps=2.0):
        assert len(vision_tensors.shape) == 4

        temporal_patch_size = 2
        merge_size = 2
        patch_size = 14

        if vision_tensors.shape[0] % temporal_patch_size != 0:
            last = vision_tensors[-1:].clone()
            vision_tensors = torch.concat([vision_tensors, last], dim=0)

        t, channel, height, width  = vision_tensors.shape
        grid_t = t // temporal_patch_size
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = vision_tensors.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=self.device, dtype=torch.int64)

        second_per_grid_ts = [temporal_patch_size / fps] * len(grid_thw)

        return flatten_patches, grid_thw, second_per_grid_ts


    def insert_vision_placeholders(self, text, grid_thw):
        merge_length = self.processor.image_processor.merge_size**2

        pad_token = self.processor.video_token if self.is_video else self.processor.image_token

        index = 0
        for i in range(len(text)):
            while pad_token in text[i]:
                text[i] = text[i].replace(
                    pad_token,
                    "<|placeholder|>" * (grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", pad_token)
        return text


    def get_vision_tensors(self, messages):
        vision_tensors = None
        if self.is_video:
            video_path = messages[0]['content'][0]['video']
            video = get_video_frames(video_path)
            nframes, c, h, w = video.shape
            resize_h, resize_w = self.smart_resize(h, w)
            video = resize_tensor(video, (resize_h, resize_w))
            normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            vision_tensors = normalize(video.div(255.))
        else:
            image_path = messages[0]['content'][0]['image']
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            resize_height, resize_width = self.smart_resize(height, width)
            vision_tensors = image_transform(image, (resize_height, resize_width)).unsqueeze(0)

        return vision_tensors.to(self.device)


    def get_model_inputs(self, messages, vision_tensors):
        pixel_values, grid_thw, second_per_grid_ts = self.process_vision(vision_tensors)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = self.insert_vision_placeholders([text], grid_thw)
        inputs = self.processor.tokenizer(text, return_tensors='pt').to(self.device)

        if self.is_video:
            inputs.update({
                'pixel_values_videos': pixel_values,
                'video_grid_thw': grid_thw,
                'second_per_grid_ts': second_per_grid_ts
            })
        else:
            inputs.update({
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw
            })
        return inputs


    def get_vision_hidden_states(self, vision_inputs):
        input_ids = vision_inputs['input_ids']
        hidden_states = vision_inputs['hidden_states']

        indices = torch.nonzero(input_ids == self.processor.video_token_id)
        video_sequence_start = indices[0][1].item()
        video_sequence_end = indices[-1][1].item()

        hidden_states = [h[:, video_sequence_start: video_sequence_end+1] for h in hidden_states]
        return hidden_states


# =========== InternVL ===========
class InternManager(VisualManager):

    def __init__(self, model, processor, is_video):
        super().__init__(model, processor, is_video)
        self.patch_size = 448


    def process_image(self, vision_tensors):
        min_patches, max_patches = 1, 12
        patch_size_height, patch_size_width = self.patch_size, self.patch_size
        original_height, original_width = vision_tensors.shape[-2:]
        
        # find the closest aspect ratio to the target
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
        )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = resize_tensor(vision_tensors, (target_height, target_width))
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = resize_tensor(vision_tensors, (patch_size_height, patch_size_width))
            processed_images.append(thumbnail_img)

        processed_images = torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()

        return processed_images[0]


    def insert_vision_placeholders(self, prompt, num_patches):
        processed_text = []
        replace_strings = []
        processor = self.processor

        pad_token = processor.video_token if self.is_video else processor.image_token
        
        while pad_token in prompt:
            if self.is_video:
                prompt = prompt.replace(pad_token, "<placeholder>", 1)
                video_prompt = "\n".join(
                    f"Frame{i + 1}: {processor.start_image_token}{processor.image_token * processor.image_seq_length}{processor.end_image_token}"
                    for i in range(num_patches)
                )
                replace_strings.append(video_prompt)
            else:
                prompt = prompt.replace(pad_token, "<placeholder>", 1)
                replace_strings.append(
                    f"{processor.start_image_token}{processor.image_token * processor.image_seq_length * num_patches}{processor.end_image_token}"
                )

        while "<placeholder>" in prompt:
            replace_str = replace_strings.pop(0)
            prompt = prompt.replace("<placeholder>", replace_str, 1)
        processed_text.append(prompt)
        return processed_text


    def get_model_inputs(self, messages, vision_tensors):
        if self.is_video:
            pixel_values = vision_tensors.to(dtype=torch.bfloat16)
        else:
            pixel_values = self.process_image(vision_tensors).to(dtype=torch.bfloat16)
    
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = self.insert_vision_placeholders(prompt, pixel_values.shape[0])
        inputs = self.processor.tokenizer(text ,return_tensors='pt').to(self.device) #, padding_side='left', return_token_type_ids=False

        inputs.update({"pixel_values": pixel_values})
            
        return inputs
    

    def get_vision_tensors(self, messages):
        vision_tensors = None
        if self.is_video:
            video_path = messages[0]['content'][0]['video']
            video = get_video_frames(video_path)
            video = resize_tensor(video, self.patch_size)
            normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            vision_tensors = normalize(video.div(255.))

        else:
            image_path = messages[0]['content'][0]['image']
            image = Image.open(image_path).convert("RGB")
            vision_tensors = image_transform(image).unsqueeze(0)

        return vision_tensors.to(self.device)


    def get_vision_hidden_states(self, vision_inputs):
        pixel_values = vision_inputs['pixel_values']
        outputs = self.model.vision_tower(pixel_values=pixel_values, output_hidden_states=True)

        return outputs.hidden_states

# =========== Llava-Next ===========
class LlavaNextManager(VisualManager):

    def __init__(self, model, processor, is_video=False):
        super().__init__(model, processor, is_video)
        self.patch_size = 336
        

    def scale_resize(self, vision_tensors, target_resolution):
        original_height, original_width = vision_tensors.shape[-2:]
        target_height, target_width = target_resolution

        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)
        return resize_tensor(vision_tensors, (new_height, new_width))
    

    def divide_to_patches(self, vision_tensors):
        patches = []
        patch_size = self.patch_size
        height, width = vision_tensors.shape[-2:]
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = vision_tensors[:, :, i : i + patch_size, j : j + patch_size]
                patches.append(patch)

        patch_tensor = resize_tensor(vision_tensors, patch_size)
        return [patch_tensor] + patches


    def process_image(self, vision_tensors):
        possible_resolutions = self.processor.image_processor.image_grid_pinpoints

        image_sizes = torch.tensor([vision_tensors.shape[-2:]])

        best_resolution = select_best_resolution(vision_tensors.shape[-2:], possible_resolutions)
        vision_tensors = self.scale_resize(vision_tensors, best_resolution)
        vision_tensors = pad_tensor_center(vision_tensors, *best_resolution)
        patches = self.divide_to_patches(vision_tensors)
        patches_tensors = torch.concat(patches).unsqueeze(0)

        return patches_tensors, image_sizes
    

    def insert_vision_placeholders(self, text, ori_size, cur_size):
        height, width = cur_size
        prompt_strings = []
        pad_token = self.processor.image_token
        for sample in text:
            while pad_token in sample:
                orig_height, orig_width = ori_size
                num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                if self.processor.vision_feature_select_strategy == "default":
                    num_image_tokens -= 1
                sample = sample.replace(pad_token, "<placeholder>" * num_image_tokens, 1)
            prompt_strings.append(sample)
        prompt_strings = [sample.replace("<placeholder>", pad_token) for sample in prompt_strings]
        return prompt_strings


    def get_model_inputs(self, messages, vision_tensors):
        
        pixel_values, image_sizes = self.process_image(vision_tensors)
    
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = self.insert_vision_placeholders([prompt], image_sizes[0].tolist(), pixel_values.shape[-2:])
        inputs = self.processor.tokenizer(text ,return_tensors='pt')

        inputs.update({
            "pixel_values": pixel_values,
            'image_sizes': image_sizes
        })
            
        return inputs.to(self.device)
    

    def get_vision_tensors(self, messages):
        
        image_path = messages[0]['content'][0]['image']
        image = Image.open(image_path).convert("RGB")
        vision_tensors = image_transform(image).unsqueeze(0)

        return vision_tensors.to(self.device)


# =========== Llava-Next-Video ===========
class LlavaNextVideoManager(VisualManager):

    def __init__(self, model, processor, is_video=True):
        super().__init__(model, processor, is_video)
    

    def insert_vision_placeholders(self, text, vision_tensors):
        num_frames, _, height, width = vision_tensors.shape
        patch_size = self.processor.patch_size
        pad_token = self.processor.video_token

        # no `self.num_additional_image_tokens` added because video always has a default feature selection strategy
        num_image_tokens = (height // patch_size) * (width // patch_size)
        num_video_tokens = num_image_tokens // 4 * num_frames  # divide by 4 needed for avg pooling layer
        prompt_strings = []
        for sample in text:
            sample = sample.replace(pad_token, pad_token * num_video_tokens)
            prompt_strings.append(sample)
        return prompt_strings
    

    def get_model_inputs(self, messages, vision_tensors):
        
        pixel_values = vision_tensors.unsqueeze(0)
    
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text = self.insert_vision_placeholders([prompt], vision_tensors)
        inputs = self.processor.tokenizer(text ,return_tensors='pt')

        inputs.update({
            "pixel_values_videos": pixel_values,
        })
            
        return inputs.to(self.device)
    

    def get_vision_tensors(self, messages):
        video_path = messages[0]['content'][0]['video']
        video = get_video_frames(video_path, return_tensors='numpy')
        videos_inputs = self.processor.video_processor(video, return_tensors='pt')
        vision_tensors = videos_inputs['pixel_values_videos'][0]

        return vision_tensors.to(self.device)


    def get_vision_hidden_states(self, vision_inputs):
        pixel_values = vision_inputs['pixel_values_videos']
        outputs = self.model.vision_tower(pixel_values=pixel_values[0], output_hidden_states=True)

        return outputs.hidden_states


# =========== InstructBLIP ===========
class InstructBLIPManager(VisualManager):

    def __init__(self, model, processor, is_video=False):
        super().__init__(model, processor, is_video)
        self.patch_size = 224
        self.eos_token_id = [1, 2]


    def get_model_inputs(self, messages, vision_tensors):
        pixel_values = vision_tensors
        prompt = messages[0]['content'][1]['text']
        inputs = self.processor(text=prompt, return_tensors="pt", images=Image.new('RGB', (224,224)))
        inputs.update({"pixel_values": pixel_values})
            
        return inputs.to(self.device)
    
    
    def get_vision_tensors(self, messages):

        image_path = messages[0]['content'][0]['image']
        image = Image.open(image_path).convert("RGB")
        vision_tensors = image_transform(image, self.patch_size).unsqueeze(0)

        return vision_tensors.to(self.device)
    

    def process_inputs(self, inputs, output_ids):
        input_ids = inputs.pop('input_ids')
        prompt = self.processor.decode(input_ids[0], skip_special_tokens=True)
        prompt += self.processor.decode(output_ids[0, input_ids.size(1):])

        qformer_text_encoding = self.processor.qformer_tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
        
        inputs.update({'qformer_input_ids': qformer_text_encoding.pop("input_ids")})
        inputs.update({'input_ids': output_ids})

        inputs.pop('attention_mask')
        inputs.pop('qformer_attention_mask')

        return inputs.to(self.device)
    
    
# =========== InstructBLIP-Video ===========
class InstructBilpVideoManager(VisualManager):

    def __init__(self, model, processor, is_video=True):
        super().__init__(model, processor, is_video)
        self.patch_size = 224
        self.nframe = 4 
        self.eos_token_id = [1, 2]


    def get_model_inputs(self, messages, vision_tensors):
        pixel_values = vision_tensors
        prompt = messages[0]['content'][1]['text']
        inputs = self.processor(text=prompt, return_tensors="pt", images=Image.new('RGB', (224,224)))
        inputs.update({"pixel_values": pixel_values})
            
        return inputs.to(self.device)
    
    
    def get_vision_tensors(self, messages):

        video_path = messages[0]['content'][0]['video']
        video = get_video_frames(video_path, nframes=self.nframe)
        video = resize_tensor(video, self.patch_size)
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        vision_tensors = normalize(video.div(255.)).unsqueeze(0)

        return vision_tensors.to(self.device)
    

    def process_inputs(self, inputs, output_ids):
        input_ids = inputs.pop('input_ids')
        prompt = self.processor.decode(input_ids[0], skip_special_tokens=True)
        prompt += self.processor.decode(output_ids[0, input_ids.size(1):])

        qformer_text_encoding = self.processor.qformer_tokenizer(prompt, add_special_tokens=True, return_tensors='pt')
        
        inputs.update({'qformer_input_ids': qformer_text_encoding.pop("input_ids")})
        inputs.update({'input_ids': output_ids})

        inputs.pop('attention_mask')
        inputs.pop('qformer_attention_mask')

        return inputs.to(self.device)


    def get_vision_hidden_states(self, vision_inputs):
        pixel_values = vision_inputs['pixel_values']
        outputs = self.model.vision_model(pixel_values=pixel_values[0], output_hidden_states=True)

        return outputs.hidden_states
    


def get_visual_manager(model_name, device='cuda', is_video=False):

    model_path = MODEL_PATHS[model_name]

    model_kwargs = {
        'device_map': device,
        'torch_dtype': torch.bfloat16,
        'low_cpu_mem_usage': True
    }

    visual_manager = None
    if "Qwen2.5-VL-7B" == model_name:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        visual_manager = QwenManager(model, processor, is_video)

    elif "InternVL3-8B" == model_name:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        visual_manager = InternManager(model, processor, is_video)

    elif "LLaVa-Next" == model_name:
        assert not is_video, 'LLaVa-Next can only handle images'
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        visual_manager = LlavaNextManager(model, processor)

    elif "LLaVa-Next-Video" == model_name:
        assert is_video, 'LLaVa-Next-Video can only handle video'
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        visual_manager = LlavaNextVideoManager(model, processor)

    elif 'InstructBLIP' == model_name:
        assert not is_video, 'InstructBLIP can only handle images'
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        processor = InstructBlipProcessor.from_pretrained(model_path)
        visual_manager = InstructBLIPManager(model, processor)

    elif 'InstructBlipVideo' == model_name:
        assert is_video, 'InstructBlipVideo can only handle video'
        model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        processor = InstructBlipVideoProcessor.from_pretrained(model_path)
        visual_manager = InstructBilpVideoManager(model, processor)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return visual_manager
