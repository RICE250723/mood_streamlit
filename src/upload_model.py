from huggingface_hub import HfApi, upload_folder

# リポジトリ名（好きな名前にしてOK）
REPO_NAME = "mood-classifier"  # 例: "mood-classifier"

# Hugging Face 上のユーザー名とリポジトリ名
REPO_ID = "RICE250727/" + REPO_NAME

# ローカルのモデルフォルダのパス（通常は 'model_output' など）
MODEL_DIR = "../model/model_output"

# 1. リポジトリを作成（すでに存在していればエラーにはなりません）
api = HfApi()
api.create_repo(repo_id=REPO_ID, private=True, exist_ok=True)

# 2. フォルダごとアップロード
upload_folder(
    folder_path=MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model"
)

print("✅ アップロード完了！Hugging Faceで確認してみてください。")