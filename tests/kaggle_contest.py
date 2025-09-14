import numpy as np
import pandas as pd
import pandas.api.types
import os
import zipfile
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy import linalg
from torchvision import transforms
import sklearn.metrics
from typing import Sequence, Union, Optional


class ParticipantVisibleError(Exception):
    pass


def download_cifar_reference_images(reference_dir: str = './cifar_reference') -> str:
    """
    Download CIFAR-10 dataset for reference images if not present.
    
    Args:
        reference_dir: Directory to store CIFAR reference images
        
    Returns:
        str: Path to directory containing reference images
    """
    if os.path.exists(os.path.join(reference_dir, 'images')) and os.listdir(os.path.join(reference_dir, 'images')):
        return os.path.join(reference_dir, 'images')
    
    os.makedirs(reference_dir, exist_ok=True)
    
    # Download pre-converted CIFAR PNG images from a safe source
    os.system("wget -q https://github.com/YoongiKim/CIFAR-10-images/archive/master.zip")
    os.system("unzip -q master.zip && mv CIFAR-10-images-master/train cifar_reference/images")
    
    return images_dir


def parse_submission_data(submission_zip_path: str):
    """
    Parse submission zip file to extract CSV data and image paths.
    
    Args:
        submission_zip_path: Path to submission.zip file
        
    Returns:
        tuple: (csv_data, images_dir) where csv_data is DataFrame and images_dir is path to images
    """
    import tempfile
    import subprocess
    
    # Create temporary directory
    extract_dir = tempfile.mkdtemp()
    
    # Use OS unzip command instead of Python zipfile
    try:
        subprocess.run(['unzip', '-q', submission_zip_path, '-d', extract_dir], check=True)
        print(f"✅ Unzipped to: {extract_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Unzip failed: {e}")
        # Fallback to Python zipfile
        with zipfile.ZipFile(submission_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    # Debug: List all extracted files
    print("Extracted files:")
    for root, dirs, files in os.walk(extract_dir):
        level = root.replace(extract_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    
    # Hardcode CSV path
    csv_path = os.path.join(extract_dir, 'submissions', 'results-dreamlayer.csv')
    print(f"Using CSV: {csv_path}")
    
    try:
        csv_data = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        csv_data = pd.read_csv(csv_path, encoding='latin-1')
    
    # Hardcode images directory
    images_dir = os.path.join(extract_dir, 'submissions', 'images')
    print(f"Using images dir: {images_dir}")
    
    return csv_data, images_dir


def compute_clip_score(csv_data: pd.DataFrame, images_dir: str) -> float:
    """
    Compute CLIP score for generated images against their prompts.
    
    Args:
        csv_data: DataFrame with prompt and image filename mappings
        images_dir: Directory containing generated images
        
    Returns:
        float: Average CLIP score across all images
    """
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    scores = []
    
    for _, row in csv_data.iterrows():
        # Get prompt and image path
        prompt = row['prompt']
        image_filename = row['filenames']  # Fixed column name
        image_path = os.path.join(images_dir, image_filename)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Compute CLIP score
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_score = logits_per_image.item()
            
        scores.append(clip_score)
    
    return np.mean(scores)


def compute_fid_score(csv_data: pd.DataFrame, images_dir: str, reference_images_dir: str) -> float:
    """
    Compute FID score between generated images and reference dataset.
    
    Args:
        csv_data: DataFrame with image filename mappings
        images_dir: Directory containing generated images
        reference_images_dir: Directory containing reference images
        
    Returns:
        float: FID score
    """
    from torchvision.models import inception_v3
    
    # Load Inception model - GET FEATURES, NOT LOGITS
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # Remove final classification layer
    inception_model.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_features(image_dir, filenames=None):
        features = []
        files = filenames if filenames else os.listdir(image_dir)
        
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0)
                
                with torch.no_grad():
                    feature = inception_model(image_tensor).squeeze().numpy()
                features.append(feature)
        
        return np.array(features)
    
    # Get features for generated images
    generated_filenames = csv_data['filenames'].tolist()  # Fixed column name
    generated_features = get_features(images_dir, generated_filenames)
    
    # Get features for reference images
    reference_features = get_features(reference_images_dir)
    
    # Calculate FID with safety checks
    if len(generated_features) == 0 or len(reference_features) == 0:
        print("⚠️ No features available")
        return 300.0
    
    mu1, sigma1 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    mu2, sigma2 = np.mean(reference_features, axis=0), np.cov(reference_features, rowvar=False)
    
    # Add small diagonal for numerical stability
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6
    
    diff = mu1 - mu2
    
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        # Check for invalid values
        if np.isnan(fid) or np.isinf(fid):
            print("FID: Invalid result, returning default FID of 100")
            return 100.0
            
    except Exception as e:
        print(f"FID: Calculation failed ({e}), returning default FID of 100")
        return 100.0
    return fid


def score(solution: pd.DataFrame, submission_zip_path: str, row_id_column_name: str, 
          labels: Optional[Sequence]=None, pos_label: Union[str, int]=1, 
          average: str='binary', weights_column_name: Optional[str]=None) -> float:
    """
    Compute final score as: 0.5 × CLIPScore - 0.5 × FID_norm
    
    Args:
        solution: Ground truth DataFrame (not used in this implementation)
        submission_zip_path: Path to submission.zip file
        row_id_column_name: Name of row ID column
        
    Returns:
        float: Final combined score
    """
    print("✅ Got submission zip path")
    
    # Parse submission data
    csv_data, images_dir = parse_submission_data(submission_zip_path)
    print(f"CSV columns: {csv_data.columns.tolist()}")
    print(f"CSV shape: {csv_data.shape}")
    print("✅ Parsed submission data")
    
    # Download CIFAR reference images if not present
    reference_images_dir = download_cifar_reference_images()
    print("✅ Downloaded CIFAR reference images")
    
    # Compute CLIP score
    clip_score = compute_clip_score(csv_data, images_dir)
    print(f"✅ Computed CLIP score: {clip_score:.4f}")
    
    # Compute FID score
    fid_score = compute_fid_score(csv_data, images_dir, reference_images_dir)
    print(f"✅ Computed FID score: {fid_score:.4f}")
    
    # Normalize FID score (assuming max FID of 300 for normalization)
    fid_norm = fid_score / 300.0
    
    # Compute final score: 0.5 × CLIPScore - 0.5 × FID_norm
    final_score = 0.5 * clip_score - 0.5 * fid_norm
    print("✅ Computed final score")
    
    return final_score
