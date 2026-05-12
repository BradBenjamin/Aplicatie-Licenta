from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import torch

cfg = get_default_cfg()
warmup_steps = 0.05 * (cfg["num_tokens"] // cfg["batch_size"]) # 5% of total steps for warmup

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

