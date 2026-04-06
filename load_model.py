import torch
from sae_lens import HookedSAETransformer, SAE
from dotenv import load_dotenv
import os
from huggingface_hub import login

def huggingface_login():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found! Check .env file.")
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace successfully!")

def load_model(model_name, sae_release, sae_id, device = "cuda"):
    '''
    Loads the LLM and the SAE. Returns both.'''
    huggingface_login() # Log in to HF
    print("Loading Model...")
    sae_model = HookedSAETransformer.from_pretrained_no_processing(
        model_name, # Assuming you switch to Gemma for the official SAEs
        device=device,
        dtype=torch.bfloat16 # Standard precision for interpretability!
    )

    # 2. Load the Pre-trained SAE
    # This example loads a Gemma Scope SAE for Layer 12
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    print(f"Loaded model: {model_name}")
    print(f"Loaded SAE: {sae_id} from release {sae_release}")
    return sae_model, sae
