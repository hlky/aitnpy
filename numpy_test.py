import math
import numpy as np
from tqdm.auto import trange, tqdm

from PIL import Image
from transformers import CLIPTokenizer
import argparse

from module import Model
from safetensors.numpy import load_file
from ckpt_convert import convert_ldm_unet_checkpoint, convert_text_enc_state_dict, convert_ldm_vae_checkpoint
from mapping import map_unet, map_vae, map_clip
from sampling import clip_inference, vae_inference, Scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="beautiful scenery nature glass bottle landscape, purple galaxy bottle")
parser.add_argument("--negative_prompt", type=str, default="low res, blurry, text")
parser.add_argument("--model_path", type=str, default="v1-5-pruned-emaonly.safetensors")
parser.add_argument("--unet_module_path", type=str, default="v1_unet_64_1024.so")
parser.add_argument("--vae_module_path", type=str, default="v1_vae_64_1024.so")
parser.add_argument("--clip_module_path", type=str, default="v1_clip.so")
parser.add_argument("--output_path", type=str, default="output.png")
args = parser.parse_args()

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")

def load_model(model_path):
    model = load_file(model_path)
    alphas_cumprod = model['alphas_cumprod']
    unet_model = convert_ldm_unet_checkpoint(model)
    unet_model = map_unet(unet_model)
    vae_model = convert_ldm_vae_checkpoint(model)
    vae_decode_model = map_vae(vae_model)
    clip_model = convert_text_enc_state_dict(model)
    clip_model = map_clip(clip_model)
    return unet_model, vae_decode_model, clip_model, alphas_cumprod

def create_model_wrapper(unet_module_path, unet_model):
    module = Model(unet_module_path)
    for k, v in unet_model.items():
        v = v.astype(np.float16)
        ait = module.numpy_to_ait_data(v)
        module.set_constant(k, ait)
    model_wrapped = Scheduler(alphas_cumprod=alphas_cumprod, quantize=True, unet_ait_exe=module)
    return model_wrapped

def text_embed(clip_module_path, clip_model, prompt, negative_prompt):
    print("tokenize text")
    clip_module = Model(clip_module_path)
    for k, v in clip_model.items():
        v = v.astype(np.float16)
        ait = clip_module.numpy_to_ait_data(v)
        clip_module.set_constant(k, ait)
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    text_embeddings = clip_inference(clip_module, text_input.input_ids)
    text_input = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    negative_text_embeddings = clip_inference(clip_module, text_input.input_ids)
    del clip_module
    return text_embeddings, negative_text_embeddings

def vae(vae_module_path, vae_model, sample):
    vae_module = Model(vae_module_path)
    for k, v in vae_model.items():
        v = v.astype(np.float16)
        ait = vae_module.numpy_to_ait_data(v)
        vae_module.set_constant(k, ait)
    sample = vae_inference(vae_module, 1. / 0.18125 * sample)
    sample = np.clip((sample + 1.0) / 2.0, a_min=0.0, a_max=1.0)
    sample = np.moveaxis(sample, 1, -1)
    sample = np.squeeze(sample, axis=0)
    image = Image.fromarray((sample * 255).astype(np.uint8))
    del vae_module
    return image

unet_model, vae_decode_model, clip_model, alphas_cumprod = load_model(args.model_path)
text_embeddings, negative_text_embeddings = text_embed(args.clip_module_path, clip_model, args.prompt, args.negative_prompt)
batch_size = 1
height = 64
width = 64
latent_channels = 4
sequence_length = 77
hidden_dim = 768
dtype = np.float16
latent_model_input = np.random.randn(batch_size, latent_channels, height, width).astype(dtype)
extra_args = {"cond":text_embeddings, "uncond":negative_text_embeddings, "cond_scale": 8.0, "cond_concat": None}
samplers = ["sample_euler_ancestral", "sample_dpm_2_ancestral", "sample_dpmpp_2s_ancestral"]
model_wrapped = create_model_wrapper(args.unet_module_path, unet_model)
for sampler in samplers:
    print(f"running {sampler}")
    sigmas = model_wrapped.get_sigmas(20)
    sample = getattr(model_wrapped, sampler)(x=latent_model_input, sigmas=sigmas, extra_args=extra_args)
    vae_image = vae(args.vae_module_path, vae_decode_model, sample)
    vae_image.save(f"{sampler}_{args.output_path}")
