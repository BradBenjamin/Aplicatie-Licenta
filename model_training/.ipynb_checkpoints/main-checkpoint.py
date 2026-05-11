from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import torch




num_tokens = 1000000000 # 1 billion ideal
l1_coeff = 0.0
warmup_steps = 0.05 * (num_tokens // get_default_cfg()["batch_size"]) # 5% of total steps for warmup
cfg = get_default_cfg()

cfg["sae_type"] = "batchtopk" # "vanilla", "topk", "batchtopk", "jumprelu"
cfg["model_name"] = "gemma-2-2b-it" # Original: "gpt2-small"
cfg["layer"] = 8
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu" # Gemini zice HuggingFaceFW/fineweb-edu inainte: "Skylion007/openwebtext"
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4 # possible cosine decay
cfg["input_unit_norm"] = True
cfg["top_k"] = 32
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
            
model = HookedTransformer.from_pretrained(
    cfg["model_name"], 
    dtype=cfg["dtype"],
    device=cfg["device"]
)


activations_store = ActivationsStore(model, cfg) # 2. Prepare activation storage
train_sae(sae, activations_store, model, cfg, warmup_steps) # 3.Train the SAE

