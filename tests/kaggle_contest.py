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
from ultralytics import YOLO
import gdown
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer


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


def download_yolo_file():
    model = YOLO('yolov8n.pt')
    return 'yolov8n.pt'

def download_from_drive(drive_link):
    import gdown
    gdown.download_folder(drive_link, output='kaggle_resources', quiet=False, use_cookies=False)
    return [os.path.join('kaggle_resources', f) for f in os.listdir('kaggle_resources') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def apply_object_detection(yolo_file, generated_df):
    model = YOLO(yolo_file)
    def detect_objects(img):
        try:
            detections = model(os.path.join('kaggle_resources', img))[0]
            return [model.names[int(cls)] for cls in detections.boxes.cls] if detections.boxes is not None else []
        except:
            return []
    generated_df['predicted_objects'] = [detect_objects(img) for img in generated_df['generated_images']]
    return generated_df

def parse_results_from_drive(drive_link):
    gdown.download_folder(drive_link, output='kaggle_resources', quiet=False, use_cookies=False)
    csv_data = pd.read_csv('kaggle_resources/results.csv')
    return csv_data[['run_id', 'prompt', 'filenames']].rename(columns={'filenames': 'generated_images'})

def parse_results_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('kaggle_resources')
    csv_data = pd.read_csv('kaggle_resources/results.csv')
    return csv_data[['run_id', 'prompt', 'filenames']].rename(columns={'filenames': 'generated_images'})

def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Calculate the average F1 score between predicted and ground truth objects using proper precision/recall.
    
    Args:
        solution: DataFrame with 'ID' and 'ground_truth' columns
        submission: DataFrame with 'ID' and 'predicted_objects' columns
    
    Returns:
        float: Average F1 score across all prompts using TP, FP, FN calculations
    """
    import ast
    merged = solution.merge(submission, on='ID', how='left')
    merged['predicted_objects'] = merged['predicted_objects'].fillna('[]').apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    merged['ground_truth'] = merged['ground_truth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    f1_scores = []
    for pred, truth in zip(merged['predicted_objects'], merged['ground_truth']):
        pred_set, truth_set = set(pred), set(truth)
        tp = len(pred_set & truth_set)
        fp = len(pred_set - truth_set)
        fn = len(truth_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def add_id_to_submission(submission_df, solution_df):
    merged = solution_df[['ID', 'prompt']].merge(submission_df, on='prompt', how='left')
    merged['predicted_objects'] = merged['predicted_objects'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
    merged[['run_id', 'generated_images']] = merged[['run_id', 'generated_images']].fillna("missing_information")
    return merged

def extract_ground_truth(drive_link):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    doc_id = drive_link.split('/d/')[1].split('/')[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    gdown.download(export_url, 'temp_prompts.txt', quiet=False)
    with open('temp_prompts.txt', 'r') as f:
        prompts = f.readlines()
    
    results = []
    for i, prompt in enumerate(prompts):
        tokens = word_tokenize(prompt.lower().strip())
        pos_tags = pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in pos_tags if pos.startswith('NN')]
        results.append({'ID': i+1, 'prompt_id': i, 'prompt': prompt.strip(), 'ground_truth': nouns, 'Usage': 'Public'})
    
    return pd.DataFrame(results)
