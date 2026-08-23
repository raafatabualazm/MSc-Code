import os
import sys
import time

from huggingface_hub import HfApi


REPO_ID = "raafatabualazm/t5gemma2-verpo-artifacts"

if len(sys.argv) != 2:
    raise SystemExit("usage: hf_upload_public_bundle.py FOLDER")

folder = os.path.abspath(sys.argv[1])
token = sys.stdin.readline().strip()
if not token:
    raise SystemExit("missing token on stdin")

api = HfApi(token=token)
api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
info = api.model_info(REPO_ID)
if info.private:
    api.update_repo_settings(repo_id=REPO_ID, private=False)

started = time.time()
result = api.upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=folder,
    commit_message=f"Upload curated public adapter bundle from {os.path.basename(folder)}",
)
print(f"UPLOAD_RESULT {result}")
print(f"UPLOAD_SECONDS {time.time() - started:.2f}")
print(f"REPO_URL https://huggingface.co/{REPO_ID}")
