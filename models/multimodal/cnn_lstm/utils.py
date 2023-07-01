import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import pandas as pd
from typing import Tuple
from torchvision.transforms import Compose, Resize, ToTensor

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class MultimodalDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, text_transform=None, image_transform=ToTensor()):
        """
        Args:
            csv_file (str): Path to the CSV file containing labels and text data.
            image_dir (str): Directory with all the images.
            text_transform (callable, optional): Optional transform to be applied to the text data.
            image_transform (callable, optional): Optional transform to be applied to the image data.
        """
        self.data_frame = pd.read_csv(csv_file, escapechar='\\')
        self.image_dir = image_dir
        self.text_transform = text_transform
        self.image_transform = Compose([Resize((224, 224)), ToTensor()])
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        def yield_tokens(data_frame):
            for _, row in data_frame.iterrows():
                yield self.tokenizer(row["text"])

        vocab = build_vocab_from_iterator(yield_tokens(self.data_frame))
        return vocab
          

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_name = os.path.join(self.image_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        
        # Apply image transformation
        if self.image_transform:
            image = self.image_transform(image)

        # Load text data
        text = self.data_frame.iloc[idx, 1]
        text = self.tokenizer(text)
        text = [self.vocab[token] for token in text]
        text = torch.tensor(text, dtype=torch.long)

        # Apply text transformation, if any
        if self.text_transform:
            text = self.text_transform(text)

        # Load label
        label = int(self.data_frame.iloc[idx, 2])

        return image, text, label
