import torch
import clip
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["Aligned_horizontally"]).to(device)

pdb.set_trace()
with torch.no_grad():
    text_features = model.encode_text(text)
    print(text_features.shape)
    
    pdb.set_trace()
  