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

# doesnt like -1 as padding token , have to set padding token to another number, will get error in embedding layer as doesnt like neg numbers 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    # Add a new padding token
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})

ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset", cache_dir="./data")['train']

PAD_TOKEN = tokenizer.pad_token
PAD_INDEX = tokenizer.pad_token_id
MAX_LENGTH = 219

print(PAD_INDEX)
from sklearn.preprocessing import LabelEncoder

# preprocess and pad, use special token for padding so can mask 
def preprocess(ds, max_records):
    texts = ds['text'] 
    labels = ds['label']

    # Create an instance of LabelEncoder
    label_encoder = LabelEncoder()

    # Fit the encoder and transform the labels
    labels = label_encoder.fit_transform(labels)

    input_ids_list = []
    attention_masks_list = []
    tensor_labels = []


    # converting zip into iterable
    for i, (text, label) in enumerate(zip(texts, labels)):
        if max_records is not None and i >= max_records:
            break

        # Tokenize the text
        encoded_inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Append the input_ids and attention_mask tensors
        input_ids_list.append(encoded_inputs['input_ids'])
        attention_masks_list.append(encoded_inputs['attention_mask'])
        tensor_labels.append(label)

    # Concatenate the lists of tensors
    input_ids = torch.cat(input_ids_list, dim=0)

    attention_masks = torch.cat(attention_masks_list, dim=0)
    labels = torch.tensor(labels[:len(input_ids)], dtype=torch.long)

    return input_ids, attention_masks, labels
    

if __name__ == '__main__':
    # Parameters
    
    save_dir = 'preprocessed_data'
    batch_size = 1
    learning_rate = 1e-4
    epochs = 1
    num_classes = 10  # Number of classes
    emb_dim = 256  # Embedding dimension
    num_heads = 2
    voc_size = 50258
    num_encoder_layers = 6  # Number of encoder layers
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device being used',device)


    # Load the preprocessed data
    input_ids, attention_masks, labels = preprocess(ds, max_records=3)
    
    # Prepare data for DataLoader
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, texts, attention_masks, labels):
            self.texts = texts
            self.attention_masks = attention_masks
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            attention_mask = self.attention_masks[idx]
            label = self.labels[idx]
            return text, attention_mask, label
        
    # Create Dataset and DataLoader
    dataset = SentimentDataset(input_ids, attention_masks, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = Encoder(
        emb_dim=emb_dim,
        num_heads=num_heads,
        vocab_size = voc_size,
        hidden_dim_ff= 64
        # num_encoder_layers=num_encoder_layers,
        # num_classes=num_classes
    ).to(device)

    wandb.init(project='semantic_model', config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    # "num_classes": num_classes,
    "emb_dim": emb_dim,
    "num_heads": num_heads,
    # "num_encoder_layers": num_encoder_layers,
    })
    
    # think about ignoring padding in my loss
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for texts, attention_masks, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            texts = texts.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            print(texts)
            
            # Adjust mel shape for Conv1d: [batch_size, channels, time_steps]
            # texts = texts.squeeze(1)  # Remove singleton dimension if present
            # mel = mel.permute(0, 2, 1)  # From [batch_size, n_mels, time_steps] to [batch_size, time_steps, n_mels]
            # mel = mel.transpose(1, 2)  # Now [batch_size, n_mels, time_steps]
            
            optimizer.zero_grad()

            # outputs = model(mel)

            # Forward pass
            class_logits = model(texts, attention_masks)

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