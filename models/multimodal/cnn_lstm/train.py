import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from cnn_lstm import cnn_lstm_classifier
from utils import MultimodalDataset
import json
# Set device
device = torch.device("mps")

# Replace these with the paths to your data
csv_file = "../../datasets/cnn_lstm.csv"
image_dir = "../../datasets/images"
classes = "../../datasets/cnn_lstm_classes.json"

# Load and preprocess the data
def collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels)
    return images, texts, labels

dataset = MultimodalDataset(csv_file, image_dir)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

with open(classes) as file:
    classes = json.load(file)

# Initialize the model and optimizer
text_vocab_size = 30522  # Set this to the size of your vocabulary
num_classes = len(classes.keys())  # Set this to the number of classes in your classification task

model = cnn_lstm_classifier(text_vocab_size=text_vocab_size, num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 128
for epoch in range(num_epochs):
    for i, (images, texts, labels) in enumerate(train_loader):
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model({"image": images, "text": texts})
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


# Testing loop
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            output = model({"image": images, "text": texts})
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test data: {accuracy:.2f}%")

# Evaluate the model on the test data
test_model(model, train_loader)


# Inference function
def infer_single_input(model, image, text):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        text = text.unsqueeze(0).to(device)
        output = model({"image": image, "text": text})
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example of using the inference function
# Replace with the index of a sample from the test dataset
sample_index = 1
image, text, label = dataset[sample_index]
predicted_class = infer_single_input(model, image, text)
print(f"Predicted class: {classes[str(predicted_class)]}, True class: {classes[str(label)]}")

torch.save(model.state_dict(), "model.pth")