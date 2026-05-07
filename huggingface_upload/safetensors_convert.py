import torch
from safetensors.torch import save_file
from huggingface_upload.huggingface_upload import GOBLIN_COLOR
def convert_pt_to_safetensors(pt_file_path, safetensors_file_path):
    # This function loads a .pt file, converts it to .safetensors, then saves both the .pt and the .safetensors to a given path
    print(f"Loading {pt_file_path}...")
    # 1. Load the PyTorch state dict. 
    # We use map_location="cpu" to ensure it loads cleanly into RAM without needing the GPU.
    tensors = torch.load(pt_file_path, map_location="cpu")

    # 2. Check if the loaded object is a dictionary (which state_dicts usually are)
    if not isinstance(tensors, dict):
        raise ValueError("The loaded .pt file is not a dictionary. Safetensors requires a dictionary of tensors.")

    # Ensure all items in the dictionary are actual tensors
    # Sometimes PyTorch saves non-tensor metadata. Safetensors only accepts tensors.
    tensors = {k: v.contiguous() for k, v in tensors.items() if isinstance(v, torch.Tensor)}

    print("Saving to safetensors format...")
    # 3. Save as safetensors
    save_file(tensors, safetensors_file_path)
    # 4. Optionally, you can also save the original .pt file to the same location for reference
    torch.save(tensors, pt_file_path.replace(".pt", f"_{GOBLIN_COLOR}_goblin.pt"))

    print(f"Success! Saved as {safetensors_file_path}")
