from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import torch


num_tokens = 250000000
l1_coeff = 0.0001 # original: 0.0018, for testing: 0.01
cfg = get_default_cfg()
cfg["sae_type"] = "batchtopk" # "vanilla", "topk", "batchtopk", "jumprelu"
cfg["model_name"] = "gemma-2-2b-it" # Original: "gpt2-small"
cfg["layer"] = 8
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "Skylion007/openwebtext"
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = True
cfg["top_k"] = 32
cfg["dict_size"] = 2304 * 8
cfg['wandb_project'] = 'batchtopk_comparison'
cfg['act_size'] = 2304
cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
cfg['bandwidth'] = 0.001
cfg['l1_coeff'] = l1_coeff
cfg['num_tokens'] = num_tokens

if cfg["sae_type"] == "vanilla":
    sae = VanillaSAE(cfg)
elif cfg["sae_type"] == "topk":
    sae = TopKSAE(cfg)
elif cfg["sae_type"] == "batchtopk":
    sae = BatchTopKSAE(cfg)
elif cfg["sae_type"] == 'jumprelu':
    sae = JumpReLUSAE(cfg)

cfg = post_init_cfg(cfg)
            
model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"]) # 1. Load model
activations_store = ActivationsStore(model, cfg) # 2. Prepare activation storage
train_sae(sae, activations_store, model, cfg) # 3.Train the SAE

'''
for sae_type in ['topk', 'batchtopk']:
    for top_k in [16, 32, 64]:
        cfg = get_default_cfg()
        cfg["sae_type"] = sae_type
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg["top_k"] = 32
        cfg["dict_size"] = 768 * 16
        cfg['wandb_project'] = 'batchtopk_comparison'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg['bandwidth'] = 0.001
        cfg['top_k'] = top_k

        if cfg["sae_type"] == "vanilla":
            sae = VanillaSAE(cfg)
        elif cfg["sae_type"] == "topk":
            sae = TopKSAE(cfg)
        elif cfg["sae_type"] == "batchtopk":
            sae = BatchTopKSAE(cfg)
        elif cfg["sae_type"] == 'jumprelu':
            sae = JumpReLUSAE(cfg)

        cfg = post_init_cfg(cfg)
                    
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)
for sae_type in ['topk', 'batchtopk']:
    # don't retrain *16
    for dict_size in [768*4, 768*8, 768*32]:
        cfg = get_default_cfg()
        cfg["sae_type"] = sae_type
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg["top_k"] = 32
        cfg["dict_size"] = dict_size
        cfg['wandb_project'] = 'batchtopk_comparison'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        cfg['bandwidth'] = 0.001
        cfg['top_k'] = 32

        if cfg["sae_type"] == "vanilla":
            sae = VanillaSAE(cfg)
        elif cfg["sae_type"] == "topk":
            sae = TopKSAE(cfg)
        elif cfg["sae_type"] == "batchtopk":
            sae = BatchTopKSAE(cfg)
        elif cfg["sae_type"] == 'jumprelu':
            sae = JumpReLUSAE(cfg)

        cfg = post_init_cfg(cfg)
                    
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)
'''