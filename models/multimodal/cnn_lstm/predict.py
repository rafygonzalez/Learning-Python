import argparse
import json
from cnn_lstm import cnn_lstm_classifier
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True,
                help="path to input image")
ap.add_argument("--caption", required=True,
                help="path to input image")
args = vars(ap.parse_args())

text_vocab_size = 30522  # Set this to the size of your vocabulary
classes = "../../datasets/cnn_lstm_classes.json"
with open(classes) as file:
    classes = json.load(file)
    
num_classes = len(classes.keys())  # Set this to the number of classes in your classification task

device  = torch.device("mps")
model = cnn_lstm_classifier(text_vocab_size=text_vocab_size, num_classes=num_classes).to(device)


    
checkpoint = torch.load("model.pth", map_location='cpu')
model.load_state_dict(checkpoint)

def infer_single_input(model, image, text):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        text = text.unsqueeze(0).to(device)
        output = model({"image": image, "text": text})
        _, predicted = torch.max(output, 1)
    return predicted.item()


tokenizer = get_tokenizer("basic_english")
image_transform = Compose([Resize((224, 224)), ToTensor()])

text = args["caption"]
image_path = args["image"]

data = [
    [image_path, args["caption"]],
]
df = pd.DataFrame(data, columns=['file_name', 'text'])

def _build_vocab(df):
    def yield_tokens(data_frame):
        for _, row in data_frame.iterrows():
            yield tokenizer(row["text"])

    vocab = build_vocab_from_iterator(yield_tokens(df))
    return vocab

text = df.iloc[0, 1]
text = tokenizer(text)
vocab = _build_vocab(df)
text = [vocab[token] for token in text]
text = torch.tensor(text, dtype=torch.long)

image = Image.open(image_path).convert("RGB")
image = image_transform(image)

predicted_class = infer_single_input(model, image, text)

print(f"Predicted class: {classes[str(predicted_class)]}")