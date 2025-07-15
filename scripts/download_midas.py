import torch
import os

# Define a predictable directory for the cache within the workspace
# This path should be relative to the repository root.
torch_hub_dir = os.path.join(os.getcwd(), 'torch_hub_cache')
os.environ['TORCH_HOME'] = torch_hub_dir
os.makedirs(torch_hub_dir, exist_ok=True)

print(f"Setting TORCH_HOME to: {torch_hub_dir}")
print("Downloading MiDaS model and transforms...")

# Load the model and transforms to trigger the download
try:
    torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
except Exception:
    # Fallback to github source if the default fails
    torch.hub.load("intel-isl/MiDaS", "MiDaS_small", source='github', trust_repo=True)
    torch.hub.load("intel-isl/MiDaS", "transforms", source='github', trust_repo=True)

print("MiDaS model downloaded successfully.")
