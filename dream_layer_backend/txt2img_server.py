from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import requests
from dream_layer import get_directories
from dream_layer_backend_utils import interrupt_workflow
from shared_utils import  send_to_comfyui
from dream_layer_backend_utils.fetch_advanced_models import get_controlnet_models
from PIL import Image, ImageDraw
from txt2img_workflow import transform_to_txt2img_workflow
from run_registry import create_run_config_from_generation_data
from dataclasses import asdict

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Get served images directory
output_dir, _ = get_directories()
SERVED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'served_images')
os.makedirs(SERVED_IMAGES_DIR, exist_ok=True)



@app.route('/api/txt2img', methods=['POST', 'OPTIONS'])
def handle_txt2img():
    """Handle text-to-image generation requests"""
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"})
    
    try:
        data = request.json
        if data:
            print("Data:", json.dumps(data, indent=2))
            
            # Print specific fields of interest
            print("\nKey Parameters:")
            print("-"*20)
            print(f"Prompt: {data.get('prompt', 'Not provided')}")
            print(f"Negative Prompt: {data.get('negative_prompt', 'Not provided')}")
            print(f"Batch Count: {data.get('batch_count', 1)}")
            
            # Check ControlNet data specifically
            controlnet_data = data.get('controlnet', {})
            print(f"\n🎮 ControlNet Data:")
            print("-"*20)
            print(f"ControlNet enabled: {controlnet_data.get('enabled', False)}")
            if controlnet_data.get('units'):
                for i, unit in enumerate(controlnet_data['units']):
                    print(f"Unit {i}:")
                    print(f"  Enabled: {unit.get('enabled', False)}")
                    print(f"  Has input_image: {unit.get('input_image') is not None}")
                    print(f"  Input image type: {type(unit.get('input_image'))}")
                    if unit.get('input_image'):
                        print(f"  Input image length: {len(unit['input_image']) if isinstance(unit['input_image'], str) else 'N/A'}")
                        print(f"  Input image preview: {unit['input_image'][:50] if isinstance(unit['input_image'], str) else 'N/A'}...")
            else:
                print("No ControlNet units found")
            
            # Extract batch_count and implement loop
            batch_count = data.get('batch_count', 1)
            all_generated_images = []
            
            for iteration in range(batch_count):
                try:
                    print(f"\n🔄 Batch iteration {iteration + 1}/{batch_count}")
                    
                    # Generate new random seed for each iteration
                    iteration_data = data.copy()
                    iteration_data['seed'] = -1  # Force new random seed
                    
                    # Transform to ComfyUI workflow
                    workflow = transform_to_txt2img_workflow(iteration_data)
                    print(f"Generated ComfyUI Workflow for iteration {iteration + 1}")
                    
                    # Send to ComfyUI server
                    comfy_response = send_to_comfyui(workflow)
                    
                    if "error" in comfy_response:
                        print(f"⚠️ Error in iteration {iteration + 1}: {comfy_response['error']}")
                        continue  # Continue with next iteration
                    
                    # Extract generated image filenames
                    generated_images = []
                    if comfy_response.get("all_images"):
                        for img_data in comfy_response["all_images"]:
                            if isinstance(img_data, dict) and "filename" in img_data:
                                generated_images.append(img_data["filename"])
                        all_generated_images.extend(comfy_response["all_images"])
                    
                    print(f"Registering run for iteration {iteration + 1}")
                    # Register the completed run - each iteration gets unique run_id
                    try:
                        from run_registry import registry
                        run_config = create_run_config_from_generation_data(
                            iteration_data, generated_images, "txt2img"
                        )
                        registry.add_run(run_config)
                        print(f"✅ Run registered with unique run_id: {run_config.run_id}")
                    except Exception as e:
                        print(f"⚠️ Error registering run for iteration {iteration + 1}: {str(e)}")
                        
                except Exception as e:
                    print(f"⚠️ Error in iteration {iteration + 1}: {str(e)}")
                    continue  # Continue with next iteration
            
            response = jsonify({
                "status": "success",
                "message": "Workflow sent to ComfyUI successfully",
                "comfy_response": {
                    "all_images": all_generated_images
                },
                "generated_images": all_generated_images
            })
            
            return response
            
        else:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400
            
    except Exception as e:
        print(f"Error in handle_txt2img: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/txt2img/batch', methods=['POST'])
def handle_txt2img_batch():
    """Handle batch text-to-image generation from prompts array"""
    try:
        data = request.json
        if not data or 'prompts' not in data:
            return jsonify({"status": "error", "message": "No prompts provided"}), 400
        
        prompts = data['prompts']
        all_generated_images = []
        failed_prompts = []
        
        for i, prompt in enumerate(prompts):
            try:
                iteration_data = data.copy()
                iteration_data['prompt'] = prompt
                
                workflow = transform_to_txt2img_workflow(iteration_data)
                comfy_response = send_to_comfyui(workflow)
                
                if "error" in comfy_response:
                    failed_prompts.append(f"Prompt {i+1}: {prompt[:50]}...")
                    continue
                
                generated_images = []
                if comfy_response.get("all_images"):
                    for img_data in comfy_response["all_images"]:
                        if isinstance(img_data, dict) and "filename" in img_data:
                            generated_images.append(img_data["filename"])
                    all_generated_images.extend(comfy_response["all_images"])
                
                from run_registry import registry
                run_config = create_run_config_from_generation_data(
                    iteration_data, generated_images, "txt2img"
                )
                registry.add_run(run_config)
                
            except Exception as e:
                failed_prompts.append(f"Prompt {i+1}: {prompt[:50]}...")
                continue
        
        return jsonify({
            "status": "success",
            "total_prompts": len(prompts),
            "processed_prompts": len(prompts) - len(failed_prompts),
            "failed_prompts": failed_prompts,
            "all_images": all_generated_images
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/txt2img/interrupt', methods=['POST'])
def handle_txt2img_interrupt():
    """Handle interruption of txt2img generation"""
    print("Interrupting txt2img generation...")
    success = interrupt_workflow()
    return jsonify({"status": "received", "interrupted": success})

@app.route('/api/images/<filename>', methods=['GET'])
def serve_image_endpoint(filename):
    """
    Serve images from multiple possible directories
    This endpoint is needed here because the frontend expects it on this port
    """
    try:
        # Use shared function
        from shared_utils import serve_image
        return serve_image(filename)
            
    except Exception as e:
        print(f"❌ Error serving image {filename}: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/controlnet/models', methods=['GET'])
def get_controlnet_models_endpoint():
    """Get available ControlNet models"""
    try:
        models = get_controlnet_models()
        return jsonify({
            "status": "success",
            "models": models
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to fetch ControlNet models: {str(e)}"
        }), 500

@app.route('/api/upload-controlnet-image', methods=['POST'])
def upload_controlnet_image_endpoint():
    """
    Endpoint to upload ControlNet images directly to ComfyUI input directory
    This endpoint is needed here because the frontend expects it on this port
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file provided"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400
        
        unit_index = request.form.get('unit_index', '0')
        try:
            unit_index = int(unit_index)
        except ValueError:
            unit_index = 0
        
        # Use shared function
        from shared_utils import upload_controlnet_image as upload_cn_image
        result = upload_cn_image(file, unit_index)
        
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
        else:
            return jsonify(result)
            
    except Exception as e:
        print(f"❌ Error uploading ControlNet image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    print("\nStarting Text2Image Handler Server...")
    print("Listening for requests at http://localhost:5001/api/txt2img")
    print("ControlNet endpoints available:")
    print("  - GET /api/controlnet/models")
    print("  - POST /api/upload-controlnet-image")
    print("  - GET /api/images/<filename>")
    app.run(host='127.0.0.1', port=5001, debug=True) 