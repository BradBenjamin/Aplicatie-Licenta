from huggingface_hub import HfApi

api = HfApi()

# 1. Create the repository on Hugging Face
repo_id = "beniaminbrad/orange_goblin_gemma" # Replace with your HF username
api.create_repo(repo_id=repo_id, exist_ok=True)

# 2. Upload your folder containing the weights and config
api.upload_folder(
    folder_path="./orange_goblin", # The local folder you created in Step 3
    repo_id=repo_id,
    repo_type="model"
)

print(f"Successfully uploaded to https://huggingface.co/{repo_id}")