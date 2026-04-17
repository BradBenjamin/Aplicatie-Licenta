import torch


def analyze_query_activations(model, sae, query, top_k=1): #Currently prints top k activations for every token in order
    
    tokens = model.to_tokens(query)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    sae_out_hook = "blocks.12.hook_resid_post.hook_sae_acts_post" # Hardcoded
    sae_acts = cache[sae_out_hook][0] 
    sentence_length = len(sae_acts)
    for index in range(sentence_length): # im hoping this iterates through the tokens of the prompt
      top_acts, top_indices = torch.topk(sae_acts[index], k=top_k)
      base_url = "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k"
      current_token_str = model.tokenizer.decode(tokens[0][index])
      print(f"Top Features for token '{current_token_str}':")
      for act, feature_idx in zip(top_acts, top_indices):
          if act.item() > 0: # Only print features that actually fired
              idx = feature_idx.item()
              activation_value = act.item()
              print(f"  Feature {idx:5d} | Activation: {activation_value:.3f} | {base_url}/{idx}")
    
def analyze_query_activations_html(model, sae, query, top_k=1):
    tokens = model.to_tokens(query)
    _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    sae_acts = cache["blocks.12.hook_resid_post.hook_sae_acts_post"][0]
    base_url = "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k"
    
    html = "<div style='font-family: monospace; font-size: 14px;'>"
    for index in range(len(sae_acts)):
        token_str = model.tokenizer.decode(tokens[0][index])
        top_acts, top_indices = torch.topk(sae_acts[index], k=top_k)
        for act, feature_idx in zip(top_acts, top_indices):
            if act.item() > 0:
                idx = feature_idx.item()
                url = f"{base_url}/{idx}"
                html += f"""
                    <div style='margin-bottom: 10px; padding: 8px; border-radius: 6px'>
                        <b>Token:</b> '{token_str}'<br>
                        <b>Feature:</b> <a href='{url}' target='_blank'>{idx}</a> 
                        | <b>Activation:</b> {act.item():.3f}
                    </div>"""
    html += "</div>"
    return html
