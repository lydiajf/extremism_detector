from pathlib import Path
import pandas as pd
import torch 
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Tokenizer
from models.encoder import Encoder
import numpy as np
import wandb

from datasets import load_dataset

# doesnt like -1 as padding token , have to set padding token to another number 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", cache_dir="./data")['train']

PAD_TOKEN = "<PAD>"
PAD_INDEX = 0
MAX_LENGTH = 219

# preprocess and pad, use special token for padding so can mask 
def preprocess(ds, max_records=None):
    texts = ds['text'] 
    labels = ds['label']

    tensor_texts = []
    tensor_labels = []

    # converting zip into iterable
    for i,(text, label) in enumerate(zip(texts, labels)):
        if max_records is not None and i >= max_records:
            break

        if len(text) <= MAX_LENGTH:
            encoded_inputs = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
            )


        # if len(text) <= MAX_LENGTH:
        #     tokenized_text = torch.tensor(encoded_inputs) 
        
        # # Pad the text to MAX_LENGTH
        # if len(tokenized_text) < MAX_LENGTH:
        #     padding = torch.full((MAX_LENGTH - len(tokenized_text),), PAD_INDEX)
        #     tokenized_text = torch.cat((tokenized_text, padding)) 
        
        tensor_texts.append(encoded_inputs)
        tensor_labels.append(label)

    return torch.stack(tensor_texts), torch.tensor(tensor_labels)
    

if __name__ == '__main__':
    # Parameters
    
    save_dir = 'preprocessed_data'
    batch_size = 32
    learning_rate = 1e-4
    epochs = 1
    num_classes = 10  # Number of classes
    emb_dim = 256  # Embedding dimension
    num_heads = 2
    hidden_dim_ff = 128
    num_encoder_layers = 6  # Number of encoder layers
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device being used',device)


    # Load the preprocessed data
    texts, labels = preprocess(ds, max_records=3)
    
    # Prepare data for DataLoader
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            # Adjust mel shape if necessary
            # Current shape: [80, time_steps]
            # For Conv1d, we need shape: [channels, time_steps]
            return text, label
    
    # Create Dataset and DataLoader
    dataset = AudioDataset(texts, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = Encoder(
        emb_dim=emb_dim,
        num_heads=num_heads,
        hidden_dim_ff=hidden_dim_ff,
        num_encoder_layers=num_encoder_layers,
        num_classes=num_classes
    ).to(device)

    wandb.init(project='semantic_model', config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_classes": num_classes,
    "emb_dim": emb_dim,
    "num_heads": num_heads,
    "hidden_dim_ff": hidden_dim_ff,
    "num_encoder_layers": num_encoder_layers,
    })
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for texts, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            texts = texts.to(device)
            labels = labels.to(device)
            
            # Adjust mel shape for Conv1d: [batch_size, channels, time_steps]
            texts = texts.squeeze(1)  # Remove singleton dimension if present
            # mel = mel.permute(0, 2, 1)  # From [batch_size, n_mels, time_steps] to [batch_size, time_steps, n_mels]
            # mel = mel.transpose(1, 2)  # Now [batch_size, n_mels, time_steps]
            
            optimizer.zero_grad()

            # outputs = model(mel)

            # Forward pass
            class_logits = model(texts)

            loss = criterion(class_logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            wandb.log({'batch_loss': loss.item(), 'epoch': epoch+1})
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/semantic_model.pth')
    # may need full path if from root
    # torch.save(model.state_dict(), 'sound_classification/models/urban_sound_model_with_splits.pth')


    ds_val = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", cache_dir="./data")['validation']
    model.eval()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        texts = ds['text'] 
        labels = ds['label']
        tensor_texts = []
        tensor_labels = []


        for text, label in zip(texts, labels):
            text = text.unsqueeze(0)  # Add batch dimension if necessary
            output = model(text)  # Forward pass
            all_preds = []
            all_labels = []

            # Assuming you have a way to get predicted class from output
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Log validation loss and accuracy to wandb
        wandb.log({'val_accuracy': accuracy, 'epoch': epoch+1})

        # Create a table of predictions and true labels
        table = wandb.Table(columns=['Predicted', 'True'])
        for pred, true_label in zip(all_preds, all_labels):
            table.add_data(pred, true_label)
        # Log the table to wandb
        wandb.log({'predictions': table})


# model.eval()

#     with torch.no_grad():  # Disable gradient calculation for evaluation
#         correct_predictions = 0
#         total_samples = 0
#         all_preds = []
#         all_labels = []

#         for mel, label in zip(spectrograms_val, labels_val):
#             mel = mel.unsqueeze(0) 
#             mel = mel.to(device) # Add batch dimension if necessary
#             labels = labels
#             output = model(mel)
#               # Forward pass

#             # Assuming you have a way to get predicted class from output
#             predicted = torch.argmax(output, dim=1)
#             correct_predictions += (predicted == label).sum().item()
#             total_samples += 1  # Increment total_samples by 1 for each label processed

#         accuracy = correct_predictions / total_samples
#         print(f"Validation Accuracy: {accuracy:.4f}")

#         # Log validation loss and accuracy to wandb
#         wandb.log({'val_accuracy': accuracy, 'epoch': epoch+1})

#         # Create a table of predictions and true labels
#         table = wandb.Table(columns=['Predicted', 'True'])
#         for pred, true_label in zip(all_preds, all_labels):
#             table.add_data(pred, true_label)
#         # Log the table to wandb
#         wandb.log({'predictions': table})