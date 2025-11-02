from huggingface_hub import HfFileSystem

fs = HfFileSystem()
try:
    files = fs.ls("datasets/pixparse/cc3m-wds", detail=True)
    print(files)
except Exception as e:
    print(f"An error occurred: {e}")
