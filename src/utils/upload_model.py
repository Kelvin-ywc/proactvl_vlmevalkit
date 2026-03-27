# 安装

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

if __name__ == "__main__":
    repo_id = "oaaoaa/AICompanion"  # 或 "your-org/my-awesome-model"

    # 如果仓库不存在则创建
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    # 上传整个文件夹（会自动用 LFS 处理大文件）
    upload_folder(
        repo_id=repo_id,
        folder_path="/home/v-weicaiyan/ds/projects/weicaiyanWorkspace/code/AICompanion/ProActMLLM/hf_cache",
        repo_type="model",
        commit_message="Upload weights and configs",
    )
    print("Done!")
