CNN-LSTM classifier, a fusion model for multimodal learning. This type of model is often used when we have different types of inputs (e.g., images and text) and we want to learn a joint representation that leverages the strengths of both modalities. Here is a breakdown of each module:

CNNEncoder: This is a Convolutional Neural Network (CNN) that serves as an image encoder. The CNN stacks multiple layers of Conv2d, MaxPool2d, and BatchNorm2d operations. It takes as input a list of dimensions for the input and output of each layer, and a list of kernel sizes for the convolution operations.

LSTMEncoder: This module is a Long Short-Term Memory (LSTM) network that works as a text encoder. This LSTM module is stacked on an embedding layer that converts the input text into dense vectors of fixed size. The LSTM takes as input the size of the vocabulary for embeddings, the size of each embedding vector, the number of features in the LSTM input and in the hidden state, and whether the LSTM is bidirectional and whether batches should be provided as (batch, seq, feature) or (seq, batch, feature).

LateFusion: This is the fusion module that combines the outputs of the image and text encoders. It uses a concatenation fusion module (ConcatFusionModule) to concatenate the outputs of the encoders, and feeds the resulting vector into an MLP (Multi-Layer Perceptron) to obtain the final output scores.

cnn_lstm_classifier: This is a builder function that creates an instance of the LateFusion classifier with a CNNEncoder for images and an LSTMEncoder for text. It takes in all parameters necessary to construct these modules, simplifying the construction of the final model. The output of the classifier is raw scores for each class.

The flexibility and composability of these modules is one of the key strengths of PyTorch, and this code showcases how these features can be leveraged to build complex models with relatively straightforward code.


# Run Inference Test

python predict.py --image ../../datasets/images/a72143c3-1acd-4955-b82b-42693891cc9a.jpg --caption "No podemos acceder a la ubicación"

Predicted class: Light Text

python predict.py --image ../../datasets/images/2b83a2fc-8f35-45c7-9c5b-729a5f5db4b2.jpg --caption "4140 Parker Rd. Allentown\, New Mexico 31134"

Predicted class: Normal Text

python predict.py --image ../../datasets/images/986e6f36-ddd3-41d6-b9a0-15ba6adfc284.jpg --caption "Enviar al domicilio"

Predicted class: Bold Text