import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd


class NameDataset(Dataset):
    def __init__(self, file):
        self.names = pd.read_csv(file)['name'].values
        self.chars = sorted(list(set(''.join(self.names) + ' ')))  # Including a padding character
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '  # Adding padding character at the end
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)


class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3) #cia didinti bloku skaiciu? 1
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0)
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x


# Load your PyTorch models
# Replace with your actual model paths and loading logic
model_man = torch.load('model_man.pt', map_location=torch.device('cpu'))
model_woman = torch.load('model_woman.pt', map_location=torch.device('cpu'))

dataset_man = NameDataset('names_man.txt')
dataset_woman = NameDataset('names_woman.txt')


def sample(model, dataset, start_str='a', max_length=20, temperature=1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension

        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)

            # Apply temperature scaling
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)

            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == ' ':  # Assume ' ' is your end-of-sequence character
                break

            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        return output_name


# Streamlit App
st.title("Lithuanian name Generator")

# Input text
input_text = st.text_input("Enter starting letters:", "")

# Model selection
model_option = st.radio("Select gender:", ("Male", "Female"))


temperature = st.slider(
    label="Adjust the temperature (higher temperature - more \"creative\" names)",
    min_value=0.1,  # Minimum value
    max_value=2.0,  # Maximum value
    value=1.0,      # Default value
    step=0.1        # Step size
)


# Generate button
if st.button("Generate name"):
    if not input_text:
        st.error("Please enter some starting letters.")
    else:
        # Select the appropriate model
        selected_model = model_man if model_option == "Male" else model_woman
        selected_dataset = dataset_man if model_option == "Female" else dataset_woman

        result = sample(selected_model, selected_dataset, start_str=input_text, temperature=temperature)
        print(result)
        st.success(f"Generated name: {result}")
