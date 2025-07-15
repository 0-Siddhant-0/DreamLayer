import torch
import numpy as np
from PIL import Image
import os
import io
import asyncio
import aiohttp
import uuid

import folder_paths
from lumaai import AsyncLumaAI

# Helper to convert a tensor to a PIL Image
def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    return Image.fromarray(image_np, 'RGB')

# Helper to convert a PIL Image to a tensor
def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class LumaPhotonDepth2Img:
    @classmethod
    def INPUT_TYPES(s):
        """Defines the input types for the node."""
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful, photorealistic image"}),
                "api_key": ("STRING", {"multiline": False}),
                "disable_depth": ("BOOLEAN", {"default": False, "label_on": "depth disabled", "label_off": "depth enabled"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_novel_view"
    CATEGORY = "DreamLayer/API"

    async def _generate_novel_view_async(self, image, prompt, api_key, disable_depth, seed, guidance_scale):
        """
        This function generates a novel-view image using the Luma Photon API, with an
        optional local depth estimation step using MiDaS.

        Args:
            image (torch.Tensor): The input image tensor.
            prompt (str): Text prompt to guide the image generation.
            api_key (str): Your Luma AI API key.
            disable_depth (bool): If True, skips the MiDaS depth estimation step. This is
                                  useful for debugging or when a depth map is not needed.
            seed (int): The seed for the generation.
            guidance_scale (float): The guidance scale for the generation.
        """
        if not disable_depth:
            print("Running MiDaS depth estimation...")
            img_np = (image.squeeze().numpy() * 255).astype(np.uint8)
            
            try:
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            except Exception:
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", source='github', trust_repo=True)
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", source='github', trust_repo=True)

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            midas.to(device)
            midas.eval()
            
            transform = midas_transforms.small_transform
            input_batch = transform(img_np).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img_np.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            
            depth_map_tensor = prediction.cpu()
            output_dir = os.path.join(folder_paths.get_output_directory(), "depth")
            os.makedirs(output_dir, exist_ok=True)
            
            depth_map_normalized = (depth_map_tensor - depth_map_tensor.min()) / (depth_map_tensor.max() - depth_map_tensor.min())
            depth_map_img = Image.fromarray((depth_map_normalized * 255).numpy().astype(np.uint8))
            
            file_path = os.path.join(output_dir, f"depth_{uuid.uuid4()}.png")
            depth_map_img.save(file_path)
            print(f"Saved depth map to {file_path}")

        print("Calling Luma Photon API...")
        client = AsyncLumaAI(api_key=api_key)
        temp_filepath = None
        try:
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = f"luma_input_{uuid.uuid4()}.png"
            temp_filepath = os.path.join(temp_dir, temp_filename)
            tensor_to_pil(image).save(temp_filepath)

            print(f"Uploading image: {temp_filepath}")
            upload_result = await client.uploads.create(file_path=temp_filepath)
            
            print("Creating generation...")
            generation = await client.generations.image.create(
                prompt=prompt, image_ref=[{"url": image_url}]
            )

            print(f"Polling generation ID: {generation.id}")
            while generation.state not in ["completed", "failed"]:
                await asyncio.sleep(5)
                generation = await client.generations.get(id=generation.id)
                print(f"Generation state: {generation.state}")

            if generation.state == "failed":
                raise Exception(f"Luma API generation failed: {generation.failure_reason}")

            result_image_url = generation.assets.image
            print(f"Downloading result from {result_image_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(result_image_url) as resp:
                    resp.raise_for_status()
                    image_data = await resp.read()
            
            result_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return (pil_to_tensor(result_image),)
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            await client.close()

    def generate_novel_view(self, image, prompt, api_key, disable_depth, seed, guidance_scale):
        return asyncio.run(self._generate_novel_view_async(image, prompt, api_key, disable_depth, seed, guidance_scale))
