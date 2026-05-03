from huggingface_hub import snapshot_download

repo_id = "enguyen1210/qwen7b_cls"

# folders_to_download = [
#     "ckd_meta_ret/*",
# ]

local_dir = snapshot_download(
    repo_id=repo_id,
    # allow_patterns=folders_to_download,
    local_dir="./meta_train",  
    repo_type="model"                
)

print(f"Folders downloaded to: {local_dir}")