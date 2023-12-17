import os
import gdown


if __name__ == '__main__':
    url = 'https://drive.google.com/uc?id=1WXg9vFu0r2YVtu4xdU7M-5tnExA8PkWn'
    output_dir = 'saved/models/final'
    os.makedirs(output_dir, exist_ok=True)
    output = 'saved/models/final/weights.pth'
    gdown.download(url, output)
