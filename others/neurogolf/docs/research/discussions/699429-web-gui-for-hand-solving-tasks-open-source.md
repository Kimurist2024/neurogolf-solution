# Web GUI for Hand Solving Tasks Open Source

- Topic ID: 699429
- URL: https://www.kaggle.com/competitions/neurogolf-2026/discussion/699429
- Author: Clark Kitchen (@clarkkitchen)
- Posted: 2026-05-13T21:37:20.169931700Z
- Votes: 15
- Total messages: 4

## Body

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F30746711%2Fc3f83a6b66eac4e11b8a101951599097%2FScreenshot%202026-05-13%20202004.png?generation=1778718021819103&alt=media)Hello,

I took all the inspiration from Chris Deotte and give him all the credit! I went ahead and built an open source Web GUI to build the ONNX by hand! I set this up like Chris suggested on a VPS and added headless agent support! You can download the Web GUI and host it yourself Github link - [https://github.com/goldbar123467/GolfWebGUI]

## Comments (4)

- **Lixin73** (2026-05-14T00:47:11.303Z, votes: {'canUpvote': True}):
  Hugging Face Upload (/api/export endpoint)
  server.py Lines 1144-1148: When you click the "Export ONNX" button, the validated ONNX model file is uploaded to a Hugging Face repository.
  
  Python
  api = HfApi(token=token)
  api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
  api.upload_file(path_or_fileobj=str(artifact), path_in_repo=remote_path, repo_id=repo_id, repo_type="model")
  What is being uploaded: The ONNX graph model you constructed (i.e., your solution).
  
  Target Repository: This is configured via the HF_REPO_ID variable in your .env file (it currently points to clarkkitchen22/neurogolf-handcrafted).
  
  Intended Behavior: This is a built-in feature of the project, not a hidden action. However, you must ensure that HF_REPO_ID is set to your own repository.
  
  [!WARNING]
  Risk: If you do not modify the HF_REPO_ID in the .env file, your solution will be uploaded to someone else's repository.

  - **Clark Kitchen** (2026-05-14T01:13:48.510Z, votes: {'canUpvote': True}):
    Thanks for flagging this. I fixed it by verifying the active HF_TOKEN owner before /api/export creates or uploads to HF_REPO_ID. This prevents a copied .env from silently uploading ONNX solutions to another Hugging Face namespace.

- **hongan** (2026-05-14T02:01:24.280Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thanks for sharing! goated

- **Lixin73** (2026-05-14T00:52:16.883Z, votes: {'totalVotes': 1, 'canUpvote': True, 'totalUpvotes': 1}):
  Thank you！
