import argparse
import math
import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download_f1')))

import av
import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
from diffusers_helper.memory import (
    DynamicSwapInstaller,
    cpu,
    fake_diffusers_current_device,
    get_cuda_free_memory_gb,
    gpu,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models,
)
from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import resize_and_center_crop, save_bcthw_as_mp4, soft_append_bcthw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the next N frames from an input image with FramePack-F1."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--frames",
        type=int,
        required=True,
        help="Number of future frames to generate after the input image.",
    )
    parser.add_argument("--prompt", type=str, default="", help="Optional motion prompt.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Optional negative prompt. Ignored when cfg is 1.0.",
    )
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--gs", type=float, default=10.0)
    parser.add_argument("--rs", type=float, default=0.0)
    parser.add_argument("--latent-window-size", type=int, default=9)
    parser.add_argument("--gpu-memory-preservation", type=float, default=6.0)
    parser.add_argument("--use-teacache", action="store_true", default=False)
    parser.add_argument("--mp4-crf", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/f1_next_frames",
        help="Directory to save the generated frames and video.",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        default=False,
        help="Also save the generated future frames as an MP4 clip.",
    )
    return parser.parse_args()


def load_models():
    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f"Free VRAM {free_mem_gb} GB")
    print(f"High-VRAM Mode: {high_vram}")

    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer"
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2"
    )
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePack_F1_I2V_HY_20250503", torch_dtype=torch.bfloat16
    ).cpu()

    for model in [vae, text_encoder, text_encoder_2, image_encoder, transformer]:
        model.eval()
        model.requires_grad_(False)

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

    return {
        "high_vram": high_vram,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "transformer": transformer,
    }


@torch.no_grad()
def generate_future_frames(args, models):
    high_vram = models["high_vram"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    vae = models["vae"]
    feature_extractor = models["feature_extractor"]
    image_encoder = models["image_encoder"]
    transformer = models["transformer"]

    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    if not high_vram:
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)

    llama_vec, clip_l_pooler = encode_prompt_conds(
        args.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    )
    if args.cfg == 1:
        llama_vec_n = torch.zeros_like(llama_vec)
        clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
            args.negative_prompt,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
        )

    from diffusers_helper.utils import crop_or_pad_yield_mask

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    input_image_np = np.array(Image.open(args.image).convert("RGB"))
    height, width = find_nearest_bucket(
        input_image_np.shape[0], input_image_np.shape[1], resolution=640
    )
    input_image_np = resize_and_center_crop(
        input_image_np, target_width=width, target_height=height
    )

    input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
    input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)
    start_latent = vae_encode(input_image_pt, vae)

    if not high_vram:
        load_model_as_complete(image_encoder, target_device=gpu)
    image_encoder_output = hf_clip_vision_encode(
        input_image_np, feature_extractor, image_encoder
    )
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

    target_total_frames = args.frames + 1
    target_total_latent_frames = int(math.ceil((target_total_frames + 3) / 4))
    latent_frames_needed = max(0, target_total_latent_frames - 1)
    total_latent_sections = max(
        1, int(math.ceil(latent_frames_needed / float(args.latent_window_size)))
    )

    print(f"Target future frames: {args.frames}")
    print(f"Generating {total_latent_sections} section(s)")

    rnd = torch.Generator("cpu").manual_seed(args.seed)

    history_latents = torch.zeros(
        size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32
    ).cpu()
    history_pixels = None
    history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
    total_generated_latent_frames = 1

    for section_index in range(total_latent_sections):
        print(f"Sampling section {section_index + 1}/{total_latent_sections}")

        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(
                transformer,
                target_device=gpu,
                preserved_memory_gb=args.gpu_memory_preservation,
            )

        if args.use_teacache:
            transformer.initialize_teacache(enable_teacache=True, num_steps=args.steps)
        else:
            transformer.initialize_teacache(enable_teacache=False)

        indices = torch.arange(
            0, sum([1, 16, 2, 1, args.latent_window_size])
        ).unsqueeze(0)
        (
            clean_latent_indices_start,
            clean_latent_4x_indices,
            clean_latent_2x_indices,
            clean_latent_1x_indices,
            latent_indices,
        ) = indices.split([1, 16, 2, 1, args.latent_window_size], dim=1)
        clean_latent_indices = torch.cat(
            [clean_latent_indices_start, clean_latent_1x_indices], dim=1
        )

        clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[
            :, :, -sum([16, 2, 1]) :, :, :
        ].split([16, 2, 1], dim=2)
        clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler="unipc",
            width=width,
            height=height,
            frames=args.latent_window_size * 4 - 3,
            real_guidance_scale=args.cfg,
            distilled_guidance_scale=args.gs,
            guidance_rescale=args.rs,
            num_inference_steps=args.steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=None,
        )

        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

        if not high_vram:
            offload_model_from_device_for_memory_preservation(
                transformer, target_device=gpu, preserved_memory_gb=8
            )
            load_model_as_complete(vae, target_device=gpu)

        real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

        if history_pixels is None:
            history_pixels = vae_decode(real_history_latents, vae).cpu()
        else:
            section_latent_frames = args.latent_window_size * 2
            overlapped_frames = args.latent_window_size * 4 - 3
            current_pixels = vae_decode(
                real_history_latents[:, :, -section_latent_frames:, :, :], vae
            ).cpu()
            history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

        if history_pixels.shape[2] >= target_total_frames:
            break

        if not high_vram:
            unload_complete_models()

    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

    if history_pixels is None:
        raise RuntimeError("No frames were decoded.")

    return history_pixels[:, :, 1 : args.frames + 1, :, :]


def save_future_frames(future_pixels, output_dir, save_mp4, mp4_crf):
    os.makedirs(output_dir, exist_ok=True)

    frames_uint8 = torch.clamp(future_pixels.float(), -1.0, 1.0) * 127.5 + 127.5
    frames_uint8 = frames_uint8.detach().cpu().to(torch.uint8)
    frames_uint8 = frames_uint8[0].permute(1, 2, 3, 0).numpy()

    for frame_index, frame in enumerate(frames_uint8, start=1):
        Image.fromarray(frame).save(
            os.path.join(output_dir, f"frame_{frame_index:04d}.png")
        )

    if save_mp4:
        video_path = os.path.join(output_dir, "future_frames.mp4")
        save_mp4_from_frames(frames_uint8, video_path, fps=30, crf=mp4_crf)
        print(f"Saved video: {video_path}")


def save_mp4_from_frames(frames_uint8, output_filename, fps=30, crf=16):
    try:
        future_pixels = (
            torch.from_numpy(frames_uint8)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .float()
            / 127.5
            - 1.0
        )
        save_bcthw_as_mp4(future_pixels, output_filename, fps=fps, crf=crf)
        return
    except (AttributeError, ImportError):
        pass

    container = av.open(output_filename, mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(frames_uint8.shape[2])
        stream.height = int(frames_uint8.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf))}

        for frame in frames_uint8:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def main():
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be a positive integer.")

    models = load_models()
    future_pixels = generate_future_frames(args, models)
    save_future_frames(future_pixels, args.output_dir, args.save_mp4, args.mp4_crf)
    print(f"Saved {future_pixels.shape[2]} future frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
