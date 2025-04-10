from huggingface_hub import upload_folder
upload_folder(
    repo_id="Aadityaramrame/carecompanion-summarizer",  # your repo
    folder_path="/Users/aditi/Desktop/CareCompanion/fine_tuned_model",  # path to model
    repo_type="model"
)
