from huggingface_hub import snapshot_download

repo_id = "minhhotboy9x/MMEB_model_grads"

# folders_to_download = [
#     "ckd_meta_ret/*",
# ]

local_dir = snapshot_download(
    repo_id=repo_id,
    # allow_patterns=folders_to_download,
    local_dir="./teacher_gradients",  
    repo_type="model"                
)

print(f"Folders downloaded to: {local_dir}")