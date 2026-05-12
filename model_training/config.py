import transformer_lens.utils as utils
import torch 

def get_default_cfg():
    default_cfg = {
        "sae_type": "batchtopk", 
        "model_name": "gemma-2-2b-it",
        "seed": 49,
        "batch_size": 4096,  
        "lr": 3e-4,
        "num_tokens": 40000000, # Initial: int(1e9), for testing: 1000000
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.999, #original: 0.99 for gpt 2
        "max_grad_norm": 1,
        "seq_len": 512,
        "dtype": torch.bfloat16, #  float16 - this is responsible for no change in training
        "site": "resid_pre",
        "layer": 8,
        "act_size": 2304,
        "dict_size": 2304 * 16, # *16 - BIG run (billion tokens)
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_batch_size": 64, 
        "num_batches_in_buffer": 128, 
        "dataset_path": "HuggingFaceFW/fineweb-edu", # original:  Skylion007/openwebtext
        "input_unit_norm": True,
        "perf_log_freq": 100,
        "checkpoint_freq": 2000,
        "n_batches_to_dead": 40, # was 5

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 128,
        "aux_penalty": (1/8),
        # for jumprelu
        "bandwidth": 0.001,
        "wandb_project": "sae_toy_tests"
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg