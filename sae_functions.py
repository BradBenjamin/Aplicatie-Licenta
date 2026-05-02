import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
import os
from sae_dashboard.data_writing_fns import save_feature_centric_vis

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
    # TODO TEMPORARY DEBUG PRINT
    print("Available Cache Keys:", cache.keys())
    sae_acts = cache["blocks.8.hook_resid_pre.hook_sae_acts_post"][0]
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

def sae_dashboard_analysis(model, sae, query, device, top_k=1, ):
    tokens = model.to_tokens(query)

    config = SaeVisConfig(
        hook_point='blocks.8.hook_resid_pre',
        features=list(range(256)), 
        minibatch_size_features=64,
        minibatch_size_tokens=256,
        device=device,
        dtype="bfloat16", 
    )

    # Generate data
    data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=tokens)

    # Save feature-centric visualization to a temporary local file
    filename = "feature_dashboard.html"
    save_feature_centric_vis(sae_vis_data=data, filename=filename)

    # Read the generated HTML file into a string
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()

    os.remove(filename)
    return html_content

import torch
def sae_dashboard_analysis_2(model, sae, query, device, top_k=3):
    # 1. Convert the input query to tokens and string tokens (for display)
    tokens = model.to_tokens(query).to(device)
    str_tokens = model.to_str_tokens(query)

    # 2. Run the model and grab the activations at the specific hook point
    # Use the hook name from your SAE config
    hook_name = 'blocks.8.hook_resid_pre'
    _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    
    # hidden_states shape: [batch_size, seq_len, d_model]
    hidden_states = cache[hook_name] 

    # 3. Pass the cached activations through the SAE to get feature activations
    # feature_acts shape: [batch_size, seq_len, d_sae]
    with torch.no_grad():
        feature_acts = sae.encode(hidden_states)
    
    # Squeeze out the batch dimension so we just have [seq_len, d_sae]
    feature_acts = feature_acts.squeeze(0)

    # 4. Find the features that fired the hardest across this specific prompt
    # Get the maximum activation value for each feature across all tokens
    max_acts_per_feature, _ = feature_acts.max(dim=0)
    
    # Get the indices of the top_k features
    top_feature_values, top_feature_indices = max_acts_per_feature.topk(top_k)

    # 5. Build the HTML output
    html_content = f"<h3 style='margin-bottom: 10px; color: #ddd;'>Top {top_k} Features Firing for this Prompt</h3>"

    for max_val, feature_idx in zip(top_feature_values, top_feature_indices):
        feature_idx = feature_idx.item()
        max_val = max_val.item()
        
        # If the max activation is 0, none of the top features fired (dead prompt)
        if max_val == 0:
            continue

        html_content += f"<div style='margin-top: 15px; color: #ddd;'><b>Feature #{feature_idx}</b> (Absolute Max Act: {max_val:.2f})</div>"
        
        # FIX 1: Added color: #000; to force text visibility. Added monospace font for neat alignment.
        html_content += "<div style='line-height: 2.5em; padding: 15px; background: #f9f9f9; color: #000; border-radius: 5px; border: 1px solid #ddd; font-family: monospace; font-size: 14px; margin-top: 5px;'>"
        
        # Get the activations for THIS specific feature across all tokens
        this_feature_acts = feature_acts[:, feature_idx]
        
        # FIX 2: Calculate max activation ignoring the <bos> token (index 0) 
        # so the rest of the sentence isn't washed out
        if len(this_feature_acts) > 1:
            max_val_for_color = this_feature_acts[1:].max().item()
            # If the rest of the sequence is dead, fallback to absolute max
            if max_val_for_color == 0: 
                max_val_for_color = max_val
        else:
            max_val_for_color = max_val

        # Build the heatmap for this feature
        for token, act in zip(str_tokens, this_feature_acts):
            val = act.item()
            
            # Normalize the alpha (opacity) based on the adjusted max
            alpha = val / max_val_for_color if max_val_for_color > 0 else 0
            # Cap at 1.0 just in case the BOS token is the one being rendered
            alpha = min(1.0, alpha) 
            
            # Use an orange color (rgb 255, 150, 0) for the highlight
            color = f"rgba(255, 150, 0, {alpha})"
            
            # FIX 3: Clean up Gemma tokens & ensure pure whitespace tokens render a background block
            clean_token = token.replace(' ', ' ').replace('<bos>', '&lt;bos&gt;')
            if not clean_token.strip():
                clean_token = "&nbsp;" * max(1, len(clean_token))
            
            # Add the highlighted token to the HTML string
            html_content += f"<span style='background-color: {color}; padding: 2px 0px; border-radius: 2px; margin: 0 1px;'>{clean_token}</span>"
            
        html_content += "</div>"

    if html_content.endswith("</h3>"):
        html_content += "<div style='color: #888;'>No features activated for this prompt.</div>"

    return html_content
