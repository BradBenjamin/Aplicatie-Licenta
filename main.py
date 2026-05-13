
#%% Cell 1 - HuggingFaceAuth & Model loading (cuda: ~ 25 sec. cpu: )
from load_model import load_model, load_feature_titles
import torch
MODEL_NAME = "google/gemma-2-2b-it"
REPOSITORY = "beniaminbrad/yellow_goblin_gemma"
FOLDER = "folder"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
sae_model, sae = load_model(MODEL_NAME, REPOSITORY, FOLDER, DEVICE) #Load the model and the SAE
feature_titles = load_feature_titles(REPOSITORY) # Load the pre-computed feature titles from Hugging Face (this is much faster than computing them on the fly every time)
#%%


#%% Cell 2 - Chat loop (~ )

import sys
from chat_functions import generate_sae_response
from sae_functions import  sae_dashboard_analysis_2, generate_feature_titles, analyze_feature_globally_gradio
import gradio as gr
print(sys.executable) # Check if env is correct
gr.close_all()

NUM_TOKENS=256
TEMP=0.7
SYSTEM_PROMPT = "You are an honest and helpful AI Assistant"
IS_SAE = True
TOP_K_ACTIVATIONS = 10 #How many activations to study
SAMPLE_DATASET = [
    "The quick brown fox jumps over the lazy dog.",
    "Gemma is an open-weights AI model developed by Google.",
    "Sparse autoencoders are fascinating tools for interpretability.",
    "I love programming in Python and PyTorch."
] * 5


def gradio_response(message, history):
    html =sae_dashboard_analysis_2(sae_model, sae, message,device=DEVICE,feature_title_dict=feature_titles, top_k=TOP_K_ACTIVATIONS) if IS_SAE else "<div style='font-family: monospace; font-size: 14px;'>SAE analysis disabled.</div>"
    if len(history) == 0:
        first_message = f"{SYSTEM_PROMPT}\n\n{message}"
        messages = [{"role": "user", "content": first_message}]
    else:
        messages = history + [{"role": "user", "content": message}]
    response = generate_sae_response(sae_model, messages, NUM_TOKENS, TEMP)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, html

def show_loading_state():
    """Immediately updates the UI to show the loading text."""
    loading_html = "<div style='font-size: 16px; font-weight: bold; color: #ff9900; padding: 10px;'>⏳ Analyzing feature globally... Please wait.</div>"
    return gr.update(value=loading_html, visible=True), gr.update(visible=False), gr.update(visible=False)

def run_global_analysis(feature_id):
    """Runs the heavy computation after the loading state is rendered."""
    if feature_id is None:
        return gr.update(value="<div style='color:red;'>Please enter a Feature ID.</div>"), gr.update(visible=False), gr.update(visible=False)
    
    # Call our adapted function
    fig, html = analyze_feature_globally_gradio(sae_model, sae, int(feature_id), SAMPLE_DATASET, DEVICE)
    
    if fig is None: # Handle case where it didn't fire
        return gr.update(value=html, visible=True), gr.update(visible=False), gr.update(visible=False)
        
    return gr.update(visible=False), gr.update(value=fig, visible=True), gr.update(value=html, visible=True)


with gr.Blocks(title="Gemma Chat") as demo:
    with gr.Row():
        # --- LEFT COLUMN: Chat and Input ---
        with gr.Column(scale=1):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type your message...")
            
        # --- RIGHT COLUMN: Activations & Global Analysis ---
        with gr.Column(scale=1):
            activation_display = gr.HTML(label="Top Activations")
            # --- NEW GLOBAL ANALYSIS UI ---
            gr.Markdown("### Global Feature Analysis")
            with gr.Row():
                feat_input = gr.Number(label="Feature ID", precision=0, scale=3)
                analyze_btn = gr.Button("Analyze Globally", scale=1, variant="primary")
                
            # These components hold the results
            global_loading_msg = gr.HTML(visible=False)
            global_plot = gr.Plot(visible=False)
            global_html = gr.HTML(visible=False)
            
    # Bind the chat submit event
    msg.submit(gradio_response, [msg, chatbot], [chatbot, activation_display])

    # Bind the Global Analysis button using event chaining (.then)
    # 1. First, show the loading message immediately
    # 2. Then, run the heavy function and update the plot/html
    analyze_btn.click(
        fn=show_loading_state, 
        inputs=[], 
        outputs=[global_loading_msg, global_plot, global_html]
    ).then(
        fn=run_global_analysis,
        inputs=[feat_input],
        outputs=[global_loading_msg, global_plot, global_html]
    )

demo.launch(
    server_name="127.0.0.1", 
    server_port=7860,        # Force 7860. If it's busy, Gradio will CRASH instead of silently hopping.
    inline=False,            # BAN Gradio from trying to render inside the VS Code Interactive Window.
    inbrowser=False,
    share=False
)
#%%