from huggingface_hub import snapshot_download

print("Downloading model files...")
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="./model_cache")
print("Download complete!")
