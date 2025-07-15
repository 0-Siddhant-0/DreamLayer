from __future__ import annotations
import torch
import numpy as np
from PIL import Image
import os
import io
import uuid
import requests

import folder_paths
from comfy_api_nodes.apis.luma_api import (
    LumaImageModel,
    LumaImageGenerationRequest,
    LumaGeneration,
    LumaModifyImageRef,
    LumaState,
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)
from comfy_api_nodes.apinode_utils import (
    upload_images_to_comfyapi,
    process_image_response,
)

# Helper to convert a PIL Image to a tensor
def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def image_result_url_extractor(response: LumaGeneration):
    return response.assets.image if hasattr(response, "assets") and hasattr(response.assets, "image") else None

class LumaPhotonDepth2Img:
    @classmethod
    def INPUT_TYPES(s):
        """Defines the input types for the node."""
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful, photorealistic image",
                    },
                ),
                "image_weight": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.98,
                        "step": 0.01,
                        "tooltip": (
                            "Weight of the image; the closer to 1.0, the less the "
                            "image will be modified."
                        ),
                    },
                ),
                "model": ([model.value for model in LumaImageModel],),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed to determine if node should re-run; actual "
                            "results are nondeterministic regardless of seed."
                        ),
                    },
                ),
                "disable_depth": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "depth disabled",
                        "label_off": "depth enabled",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "depth_map_image")
    FUNCTION = "generate_novel_view"
    CATEGORY = "DreamLayer/API"

    def generate_novel_view(
        self,
        image: torch.Tensor,
        prompt: str,
        image_weight: float,
        model: str,
        seed: int,
        disable_depth: bool,
        unique_id: str = None,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Generates a novel-view image using the Luma Photon API.

        This node can optionally run a local MiDaS model to estimate a depth map
        from the input image.

        Args:
            image (torch.Tensor): The input image tensor.
            prompt (str): Text prompt to guide the image generation.
            image_weight (float): The weight of the input image.
            model (str): The Luma model to use for generation.
            seed (int): The seed for generation.
            disable_depth (bool): If True, skips MiDaS depth estimation. This is
                useful for debugging or when a depth map is not needed.
            unique_id (str, optional): The unique ID of the node.
            **kwargs: Additional keyword arguments for authentication.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor]: A tuple containing the
                generated image and the depth map image. The generated image can
                be None if the API call is skipped.
        """
        depth_map_to_return = torch.zeros_like(image)
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
                    prediction.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map_tensor = prediction.cpu()
            output_dir = os.path.join(folder_paths.get_output_directory(), "depth")
            os.makedirs(output_dir, exist_ok=True)
            
            depth_map_normalized = (depth_map_tensor - depth_map_tensor.min()) / (
                depth_map_tensor.max() - depth_map_tensor.min()
            )
            depth_map_img = Image.fromarray(
                (depth_map_normalized * 255).numpy().astype(np.uint8)
            )
            
            file_path = os.path.join(output_dir, f"depth_{uuid.uuid4()}.png")
            depth_map_img.save(file_path)
            print(f"Saved depth map to {file_path}")
            depth_map_to_return = pil_to_tensor(depth_map_img.convert("RGB"))

        if not kwargs.get("comfy_api_key"):
            print("No API key provided. Skipping Luma API call and returning depth map.")
            return (None, depth_map_to_return)

        print("Calling Luma Photon API...")
        auth_kwargs = kwargs
        
        download_urls = upload_images_to_comfyapi(
            image, max_images=1, auth_kwargs=auth_kwargs
        )
        image_url = download_urls[0]

        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/luma/generations/image",
                method=HttpMethod.POST,
                request_model=LumaImageGenerationRequest,
                response_model=LumaGeneration,
            ),
            request=LumaImageGenerationRequest(
                prompt=prompt,
                model=model,
                modify_image_ref=LumaModifyImageRef(
                    url=image_url,
                    weight=round(max(min(1.0 - image_weight, 0.98), 0.0), 2),
                ),
            ),
            auth_kwargs=auth_kwargs,
        )
        response_api: LumaGeneration = operation.execute()

        operation = PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"/proxy/luma/generations/{response_api.id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=LumaGeneration,
            ),
            completed_statuses=[LumaState.completed],
            failed_statuses=[LumaState.failed],
            status_extractor=lambda x: x.state,
            result_url_extractor=image_result_url_extractor,
            node_id=unique_id,
            auth_kwargs=auth_kwargs,
        )
        response_poll = operation.execute()

        img_response = requests.get(response_poll.assets.image)
        img = process_image_response(img_response)
        return (img, depth_map_to_return)
