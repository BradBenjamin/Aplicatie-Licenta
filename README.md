# Sparse Autoencoder Training and Evaluation
## Model Training

1. The full code for training of a SAE is in the folder: model_training and is called with: python main.py
2. Training parameters are to be changed in: config.py
3. For specific models, like Gemma, permission is needed from huggingface so if this is the case, you should make an account, and the code will ask you to log in.
4. The evaluation charts will be displayed in WANDB.

## Uploading the model to Huggingface

After training a model, I wrote some code to directly upload it to huggingface using these steps:
1. Dump sae.pt and config.json from checkpoints into huggingface_upload
2. Modify GOBLIN_COLOR in huggingface_upload.py
3. Run: python huggingface_upload.py

## Talking to the model

This logic is provided in the main directory in the form of jupyter cells. In main.py there are 2 Jupyter cells that do the following:
The first one, when called, loads the model and its respective autoencoder from huggingface
The second one starts a gradio UI that enables the user to talk to the model and see the real time token activations.
