import json
import os

def convert_sae_config(input_filepath="config.json", output_filepath="cfg.json"):
    # Check if the file exists before trying to open it
    if not os.path.exists(input_filepath):
        print(f"Error: Could not find '{input_filepath}'. Make sure it is in the same folder as this script.")
        return

    # Read the original JSON file
    with open(input_filepath, 'r') as f:
        old_cfg = json.load(f)
    
    # Map old keys to the new Hugging Face compatible structure
    new_cfg = {
        "d_in": old_cfg.get("act_size"),
        "d_sae": old_cfg.get("dict_size"),
        "architecture": "batch_topk" if old_cfg.get("sae_type") == "batchtopk" else old_cfg.get("sae_type"),
        "model_name": old_cfg.get("model_name"),
        "hook_name": old_cfg.get("hook_point"),
        "hook_layer": old_cfg.get("layer"),
        "context_size": old_cfg.get("seq_len"),
        # Strip 'torch.' from the dtype string to match HF format
        "dtype": old_cfg.get("dtype", "").replace("torch.", ""),
        "device": old_cfg.get("device"),
        "activation_fn_str": "topk",
        "activation_fn_kwargs": {
            "k": old_cfg.get("top_k")
        },
        "dataset_path": old_cfg.get("dataset_path"),
        
        # Hardcoded default parameters
        "prepend_bos": True,
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False
    }

    # Write the transformed config to the new file
    with open(output_filepath, 'w') as f:
        json.dump(new_cfg, f, indent=2)
    
    print(f"Successfully converted '{input_filepath}' and saved to '{output_filepath}'")

# --- Usage ---
if __name__ == "__main__":
    # The script now automatically looks for 'config.json' and outputs 'cfg.json'
    convert_sae_config()