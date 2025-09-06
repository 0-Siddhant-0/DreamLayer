"""
Banana (Gemini 2.5 Flash Image) Node for ComfyUI
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from inspect import cleandoc
from comfy.comfy_types.node_typing import ComfyNodeABC, IO
print("🍌 BANANA: ComfyNodeABC imported successfully")

print("🍌 BANANA MODULE: nodes_banana.py is being imported!")


class BananaImageNode(ComfyNodeABC):
    """
    Generate and edit images using Google's Gemini 2.5 Flash Image (Nano Banana) model.
    Requires paid Gemini API access.
    """
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "api_call"
    API_NODE = True
    CATEGORY = "api node/image/Banana (Gemini)"
    DESCRIPTION = cleandoc(__doc__ or "")

    @classmethod
    def INPUT_TYPES(s):
        print("🍌 BANANA: INPUT_TYPES method called!")
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Generate a high-quality, photorealistic image",
                    "multiline": True,
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Choose the type of image operation"
                }),
            },
            "optional": {
                "reference_image_1": (IO.IMAGE, {
                    "forceInput": False,
                    "tooltip": "Primary reference image for editing/style transfer"
                }),
                "reference_image_2": (IO.IMAGE, {
                    "forceInput": False,
                    "tooltip": "Second reference image (optional)"
                }),
                "reference_image_3": (IO.IMAGE, {
                    "forceInput": False,
                    "tooltip": "Third reference image (optional)"
                }),
                "batch_count": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "Number of images to generate (costs multiply)"
                }),
                "temperature": (IO.FLOAT, {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Creativity level (0.0 = deterministic, 1.0 = very creative)"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {
                    "default": "1:1",
                    "tooltip": "Output image aspect ratio"
                }),
                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Image generation quality"
                }),
            },
            "hidden": {
                "gemini_api_key": "GEMINI_API_KEY",
            },
        }

    def __init__(self):
        self.api_key = None

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(image_np, mode='RGB')

    def create_placeholder_image(self, width=512, height=512):
        """Create a placeholder image when generation fails"""
        img = Image.new('RGB', (width, height), color=(100, 100, 100))
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((width//2-50, height//2), "Generation\nFailed", fill=(255, 255, 255))
        except:
            pass
        image_array = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_array).unsqueeze(0)

    def prepare_images_for_api(self, img1=None, img2=None, img3=None):
        """Convert up to 3 tensor images to base64 format for API"""
        encoded_images = []
        for i, img in enumerate([img1, img2, img3], 1):
            if img is not None:
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 4:
                        pil_image = self.tensor_to_image(img[0])
                    else:
                        pil_image = self.tensor_to_image(img)
                    encoded_images.append(self._image_to_base64(pil_image))
        return encoded_images

    def _image_to_base64(self, pil_image):
        """Convert PIL image to base64 format for API"""
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(img_bytes).decode('utf-8')
            }
        }

    def build_prompt_for_operation(self, prompt, operation, has_references=False, aspect_ratio="1:1"):
        """Build optimized prompt based on operation type"""
        aspect_instructions = {
            "1:1": "square format",
            "16:9": "widescreen landscape format",
            "9:16": "portrait format",
            "4:3": "standard landscape format",
            "3:4": "standard portrait format"
        }

        base_quality = "Generate a high-quality, photorealistic image"
        format_instruction = f"in {aspect_instructions.get(aspect_ratio, 'square format')}"

        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}. {format_instruction}."
            else:
                final_prompt = f"{base_quality} of: {prompt}. {format_instruction}."
        elif operation == "edit":
            if not has_references:
                return "Error: Edit operation requires reference images"
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."
        elif operation == "style_transfer":
            if not has_references:
                return "Error: Style transfer requires reference images"
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally. {format_instruction}."
        elif operation == "object_insertion":
            if not has_references:
                return "Error: Object insertion requires reference images"
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective. {format_instruction}."

        return final_prompt

    def call_gemini_api(self, prompt, encoded_images, temperature, batch_count):
        """Make API call to Gemini 2.5 Flash Image"""
        try:
            import google.generativeai as genai
            
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Create client and get model
            client = genai.Client()
            model = client.get_model("gemini-2.0-flash-exp")
            
            all_generated_images = []
            operation_log = ""
            
            for i in range(batch_count):
                try:
                    # Create content parts
                    content_parts = [genai.types.TextPart(prompt)]
                    
                    # Add reference images correctly
                    for img_data_dict in encoded_images:
                        try:
                            # Decode base64 to bytes
                            image_bytes = base64.b64decode(img_data_dict["inline_data"]["data"])
                            # Convert bytes to PIL Image first
                            pil_image = Image.open(BytesIO(image_bytes))
                            # Now pass PIL Image to ImagePart
                            content_parts.append(genai.types.ImagePart(pil_image))
                        except Exception as img_prep_error:
                            operation_log += f"Error preparing reference image: {str(img_prep_error)}\n"
                            continue
                    
                    # Generate content with image response modality
                    response = model.generate_content(
                        content_parts,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            response_modalities=['Text', 'Image']
                        )
                    )
                    
                    batch_images = []
                    response_text = ""
                    
                    # Process response
                    for candidate in response.candidates:
                        for part in candidate.content.parts:
                            if hasattr(part, 'image') and part.image:
                                try:
                                    # part.image is already a PIL Image object
                                    pil_image = part.image
                                    # Convert PIL Image to bytes for tensor processing
                                    img_byte_arr = BytesIO()
                                    pil_image.save(img_byte_arr, format='PNG')
                                    all_generated_images.append(img_byte_arr.getvalue())
                                except Exception as img_error:
                                    operation_log += f"Error extracting image: {str(img_error)}\n"
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text + "\n"
                    
                    if batch_images:
                        all_generated_images.extend(batch_images)
                        operation_log += f"Batch {i+1}: Generated {len(batch_images)} images\n"
                    else:
                        operation_log += f"Batch {i+1}: No images found. Text: {response_text[:100]}...\n"
                        
                except Exception as batch_error:
                    operation_log += f"Batch {i+1} error: {str(batch_error)}\n"
            
            generated_tensors = []
            if all_generated_images:
                for img_data in all_generated_images:
                    try:
                        # Convert image data to PIL Image and then to tensor
                        if hasattr(img_data, 'data'):
                            image_bytes = img_data.data
                        else:
                            image_bytes = img_data
                        
                        image = Image.open(BytesIO(image_bytes))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        img_np = np.array(image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np)[None,]
                        generated_tensors.append(img_tensor)
                    except Exception as e:
                        operation_log += f"Error processing image: {e}\n"
            
            return generated_tensors, operation_log
            
        except ImportError:
            operation_log = "google-generativeai not available. Please install: pip install google-generativeai\n"
            return [], operation_log
        except Exception as e:
            operation_log = f"Error in Gemini API call: {str(e)}\n"
            return [], operation_log

    def api_call(self, prompt, operation, reference_image_1=None, **kwargs):
        print(f"🍌 BANANA: Node called with prompt='{prompt}'")
        
        # Get API key from kwargs or environment
        self.api_key = kwargs.get('gemini_api_key') or kwargs.get('api_key_comfy_org') or os.environ.get('GEMINI_API_KEY')
        
        if not self.api_key:
            print(f"🍌 BANANA (Gemini): ERROR - No API key provided")
            error_msg = "BANANA ERROR: No Gemini API key provided!\n\n"
            error_msg += "Gemini 2.5 Flash Image requires a PAID API key.\n"
            error_msg += "Get yours at: https://aistudio.google.com/app/apikey\n"
            error_msg += "Note: Free tier users cannot access image generation models."
            return (self.create_placeholder_image(), error_msg)

        try:
            # Process reference images
            encoded_images = self.prepare_images_for_api(
                reference_image_1, reference_image_2, reference_image_3
            )
            has_references = len(encoded_images) > 0

            # Build optimized prompt
            final_prompt = self.build_prompt_for_operation(
                prompt, operation, has_references, aspect_ratio
            )
            
            if "Error:" in final_prompt:
                return (self.create_placeholder_image(), final_prompt)

            # Add quality instructions
            if quality == "high":
                final_prompt += " Use the highest quality settings available."

            # Log operation start
            operation_log = f"BANANA OPERATION LOG\n"
            operation_log += f"Operation: {operation.upper()}\n"
            operation_log += f"Reference Images: {len(encoded_images)}\n"
            operation_log += f"Batch Count: {batch_count}\n"
            operation_log += f"Temperature: {temperature}\n"
            operation_log += f"Quality: {quality}\n"
            operation_log += f"Aspect Ratio: {aspect_ratio}\n"
            operation_log += f"Note: Output resolution determined by API (max ~1024px)\n"
            operation_log += f"Prompt: {final_prompt[:150]}...\n\n"

            # Make API call
            generated_images, api_log = self.call_gemini_api(
                final_prompt, encoded_images, temperature, batch_count
            )
            operation_log += api_log

            # Process results
            if generated_images:
                combined_tensor = torch.cat(generated_images, dim=0)
                approx_cost = len(generated_images) * 0.039  # ~$0.039 per image
                operation_log += f"\nEstimated cost: ~${approx_cost:.3f}\n"
                operation_log += f"Successfully generated {len(generated_images)} image(s)!"
                return (combined_tensor, operation_log)
            else:
                operation_log += "\nNo images were generated. Check the log above for details."
                return (self.create_placeholder_image(), operation_log)

        except Exception as e:
            print(f"🍌 BANANA (Gemini): ERROR - {str(e)}")
            error_log = f"BANANA ERROR: {str(e)}\n"
            error_log += "Please check your API key, internet connection, and paid tier status."
            return (self.create_placeholder_image(), error_log)


# Node registration
print("🍌 BANANA: Registering BananaImageNode in mappings")
NODE_CLASS_MAPPINGS = {
    "BananaImageNode": BananaImageNode,
}

# Manual merge into global registry (5-line fix)
try:
    import nodes
    nodes.NODE_CLASS_MAPPINGS["BananaImageNode"] = BananaImageNode
    print("🍌 BANANA: ✅ Manually merged into global registry")
except Exception as e:
    print(f"🍌 BANANA: Manual merge failed: {e}")

print(f"🍌 BANANA: Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaImageNode": "Banana (Gemini 2.5 Flash Image)",
}
