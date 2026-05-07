from huggingface_hub import HfApi
from json_mason import convert_sae_config
import torch
from safetensors.torch import save_file
api = HfApi()

GOBLIN_COLOR = "yellow"

def convert_pt_to_safetensors():
    folder_path = f"{GOBLIN_COLOR}_goblin"
    os.makedirs(folder_path, exist_ok=True)
    
    pt_file_path = "sae.pt"
    safetensors_file_path = os.path.join(folder_path, "sae_weights.safetensors")
    target_pt_path = os.path.join(folder_path, "sae.pt")
    
    print(f"Loading {pt_file_path}...")
    tensors = torch.load(pt_file_path, map_location="cpu")
    
    if not isinstance(tensors, dict):
        raise ValueError("The loaded .pt file is not a dictionary. Safetensors requires a dictionary of tensors.")
        
    tensors = {k: v.contiguous() for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    
    print("Saving to safetensors format...")
    save_file(tensors, safetensors_file_path)
    torch.save(tensors, target_pt_path)
    
    print(f"Success! Saved both files to {folder_path}/")

#0. Create the local folder to add the files
import os
folder_name = f"{GOBLIN_COLOR}_goblin"
os.makedirs(folder_name, exist_ok=True)
#1. Modify config using the conversion function
convert_sae_config(input_filepath="config.json", output_filepath=f"{folder_name}/cfg.json")
#2. Convert the weights to safetensors format and save both to the folder
convert_pt_to_safetensors() # This also saves both to folder
#3. Inside the folder, create a new folder named "folder"
os.makedirs(f"{folder_name}/folder", exist_ok=True)
#4. Move the cfg.json and sae_weights.safetensors into the "folder"
os.rename(f"{folder_name}/cfg.json", f"{folder_name}/folder/cfg.json")
os.rename(f"{folder_name}/sae_weights.safetensors", f"{folder_name}/folder/sae_weights.safetensors")
#5. Delete original .pt and .json files from the main directory (optional cleanup)
os.remove("config.json")
os.remove("sae.pt")

#Upload all to HF
repo_id = f"beniaminbrad/{GOBLIN_COLOR}_goblin_gemma" # Replace with your HF username
api.create_repo(repo_id=repo_id, exist_ok=True)
api.upload_folder(
    folder_path=f"./{GOBLIN_COLOR}_goblin", # The local folder you created in Step 3
    repo_id=repo_id,
    repo_type="model"
)
print(f"Successfully uploaded to https://huggingface.co/{repo_id}")