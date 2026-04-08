

#%% Cell 1 - HuggingFaceAuth & Model loading (~ 3min)

%load_ext autoreload
%autoreload 2
from load_model import load_model

#Model Loading
MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
DEVICE = "cuda"

sae_model, sae = load_model(MODEL_NAME, SAE_RELEASE, SAE_ID, DEVICE) #Load the model and the SAE
#%%
``
#%% Cell 2 - Chat loop (~ )
from chat_functions import chat_loop
NUM_TOKENS=256
TEMP=0.7
SYSTEM_PROMPT = "You are an honest and helpful AI Assistant"
IS_SAE = True
TOP_K_ACTIVATIONS = 1 #How many activations to study

chat_loop(sae_model, sae_model.tokenizer, SYSTEM_PROMPT, NUM_TOKENS, TEMP, IS_SAE, sae, TOP_K_ACTIVATIONS)
#%%