import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def is_mri_clip(the_image):

    image = preprocess(the_image).unsqueeze(0).to(device)

    texts = clip.tokenize([
        "an MRI scan of a human brain",
        "something else"
    ]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)

        logits_per_image, _ = model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    labels = [
        "MRI brain scan ",
        "something else"
    ]
    if probs[0][0] > 0.5:
        return True
    else:
        return False


def main():
    pass

if __name__ == "__main__":
    main()