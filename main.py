
#%% Cell 1 - HuggingFaceAuth & Model loading (~ 25 sec)
from load_model import load_model
MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
DEVICE = "cuda"

sae_model, sae = load_model(MODEL_NAME, SAE_RELEASE, SAE_ID, DEVICE) #Load the model and the SAE
#%%


#%% Cell 2 - Chat loop (~ )
from chat_functions import generate_sae_response
from sae_functions import analyze_query_activations_html
import gradio as gr
gr.close_all()


NUM_TOKENS=256
TEMP=0.7
SYSTEM_PROMPT = "You are an honest and helpful AI Assistant"
IS_SAE = True
TOP_K_ACTIVATIONS = 1 #How many activations to study

def gradio_response(message, history):
    html = analyze_query_activations_html(sae_model, sae, message, TOP_K_ACTIVATIONS) if IS_SAE else ""
    if len(history) == 0:
        first_message = f"{SYSTEM_PROMPT}\n\n{message}"
        messages = [{"role": "user", "content": first_message}]
    else:
        messages = history + [{"role": "user", "content": message}]
    response = generate_sae_response(sae_model, messages, NUM_TOKENS, TEMP)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, html

with gr.Blocks(title="Gemma-2-2b-it Chat") as demo:
    with gr.Row():
        # --- LEFT COLUMN: Chat and Input ---
        with gr.Column(scale=1):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Type your message...")
            
        # --- RIGHT COLUMN: Activations ---
        with gr.Column(scale=1):
            activation_display = gr.HTML(label="Top Activations")
            
    # Bind the submit event down here, AFTER all components are defined
    msg.submit(gradio_response, [msg, chatbot], [chatbot, activation_display])

demo.launch() #run interface
#%%