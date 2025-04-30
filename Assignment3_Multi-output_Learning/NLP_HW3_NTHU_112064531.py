import os
import numpy as np
import transformers as T
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef, Accuracy, F1Score
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/", trust_remote_code=True
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
lr = 3e-5
epochs = 20
train_batch_size = 16
validation_batch_size = 8
ckpt_path = '/workspace/NLP/HW3/saved_models'
ckpt_name = f'batch_{train_batch_size}_p5_clamp_best.ckpt'
plot_name = 'combined_plot_p5_clamp.png'

# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Write your code here
        # Define what modules you will use in the model
        self.bert = T.BertModel.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")
        self.fc_common = torch.nn.Linear(768, 768)
        self.fc_relate = torch.nn.Linear(768, 1)
        self.fc_entailment = torch.nn.Linear(768, 3)
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, **kwargs):
        # Write your code here
        # Forward pass
        outputs = self.bert(**kwargs)
        pooled_output = outputs.pooler_output  # shape (batch_size, hidden_size)
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = self.fc_common(pooled_output)
        relate = self.fc_relate(pooled_output).squeeze(-1)  # shape (batch_size,)
        entailment = self.fc_entailment(pooled_output)  # shape (batch_size, 3)
        return relate, entailment

model = MultiLabelModel().to(device)
tokenizer = T.BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

tokenizer(data_sample[0]['premise'])['input_ids']

# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.



def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premises = [data['premise'] for data in batch]
    hypotheses = [data['hypothesis'] for data in batch]
    inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors='pt')
    relatedness_scores = torch.tensor([data['relatedness_score'] for data in batch], dtype=torch.float)
    entailment_judgments = torch.tensor([data['entailment_judgment'] for data in batch], dtype=torch.long)
    return inputs, relatedness_scores, entailment_judgments

# TODO1-2: Define your DataLoader
dl_train = DataLoader(SemevalDataset(split="train"),
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn)
dl_validation = DataLoader(SemevalDataset(split="validation"),
            batch_size=validation_batch_size,
            shuffle=False,
            collate_fn=collate_fn)

dl_test = DataLoader(SemevalDataset(split="test"),
            batch_size=validation_batch_size,
            shuffle=False,
            collate_fn=collate_fn)

# TODO3: Define your optimizer and loss function
# os.makedirs('./saved_models', exist_ok=True)
# TODO3-1: Define your Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


class FocalLoss(nn. Module):
    def __init__(self, gamma=2, weight=None):
        super (FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss (weight=self.weight) (inputs, targets)
        pt = torch.exp(-ce_loss) 
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss
    
# TODO3-2: Define your loss functions (you should have two)
# Write your code here
relate_criterion = torch.nn.MSELoss()
weights = torch.tensor([0.2, 0.30, 0.5], dtype=torch.float32).to(device)
# Using focal loss
# entailment_criterion = FocalLoss(weight=weights)
# Using CrossEntropyLoss
entailment_criterion = torch.nn.CrossEntropyLoss(weight=weights)
# scoring functions
pearson = PearsonCorrCoef()
spc = SpearmanCorrCoef()
acc = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3, average='macro')

train_pearson_corr_list = []
valid_pearson_corr_list = []
train_spearman_corr_list = []
valid_spearman_corr_list = []
train_accuracy_list = []
valid_accuracy_list = []
train_f1_list = []
valid_f1_list = []
best_performance = 0.0

for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model

    train_relatedness_preds = []
    train_relatedness_labels = []
    train_entailment_preds = []
    train_entailment_labels = []
    for inputs, relatedness_scores, entailment_judgments in pbar:

        inputs = {k: v.to(device) for k, v in inputs.items()}
        relatedness_scores = relatedness_scores.to(device)
        entailment_judgments = entailment_judgments.to(device)

        # clear gradient
        optimizer.zero_grad()
        # forward pass

        relate_pred, entailment_pred = model(**inputs)

        train_relatedness_preds.append(relate_pred.cpu())
        train_relatedness_labels.append(relatedness_scores.cpu())
        train_entailment_preds.append(entailment_pred.cpu())
        train_entailment_labels.append(entailment_judgments.cpu())
        # compute loss
        relate_loss = relate_criterion(relate_pred, relatedness_scores)
        entailment_loss = entailment_criterion(entailment_pred, entailment_judgments)

        # back-propagation and optimization
        loss = relate_loss + entailment_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.fc_relate.parameters(), 2) # gradient clipping
        
        optimizer.step()

        pbar.set_postfix({"relate_loss": relate_loss.item(), "entailment_loss": entailment_loss.item()})

    train_relatedness_preds = torch.cat(train_relatedness_preds)
    train_relatedness_labels = torch.cat(train_relatedness_labels)
    train_entailment_preds = torch.cat(train_entailment_preds)
    train_entailment_labels = torch.cat(train_entailment_labels)

    # Compute Spearman correlation
    train_pearson_corr = pearson(train_relatedness_preds, train_relatedness_labels)
    train_spearman_corr = spc(train_relatedness_preds, train_relatedness_labels)

    # Compute accuracy and F1 score
    entailment_pred_classes = torch.argmax(train_entailment_preds, dim=1)
    acc = Accuracy(task="multiclass", num_classes=3)
    f1 = F1Score(task="multiclass", num_classes=3, average='macro')

    train_accuracy_score = acc(entailment_pred_classes, train_entailment_labels)
    train_f1_score = f1(entailment_pred_classes, train_entailment_labels)

    train_pearson_corr_list.append(train_pearson_corr.detach().numpy())
    train_spearman_corr_list.append(train_spearman_corr.detach().numpy())
    train_accuracy_list.append(train_accuracy_score.detach().numpy())
    train_f1_list.append(train_f1_score.detach().numpy())


    # Evaluation Phase
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()

    all_relatedness_preds = []
    all_relatedness_labels = []
    all_entailment_preds = []
    all_entailment_labels = []
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)

    with torch.no_grad():
        for inputs, relatedness_scores, entailment_judgments in pbar:
            # Move data to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            relatedness_scores = relatedness_scores.to(device)
            entailment_judgments = entailment_judgments.to(device)
            # Forward pass
            relate_pred, entailment_pred = model(**inputs)

            # Collect predictions and labels
            all_relatedness_preds.append(relate_pred.cpu())
            all_relatedness_labels.append(relatedness_scores.cpu())
            all_entailment_preds.append(entailment_pred.cpu())
            all_entailment_labels.append(entailment_judgments.cpu())

        all_relatedness_preds = torch.cat(all_relatedness_preds)
        all_relatedness_labels = torch.cat(all_relatedness_labels)
        all_entailment_preds = torch.cat(all_entailment_preds)
        all_entailment_labels = torch.cat(all_entailment_labels)

        # Compute Spearman correlation
        valid_pearson_corr = pearson(all_relatedness_preds, all_relatedness_labels)
        valid_spearman_corr = spc(all_relatedness_preds, all_relatedness_labels)

        # Compute accuracy and F1 score
        entailment_pred_classes = torch.argmax(all_entailment_preds, dim=1)
        acc = Accuracy(task="multiclass", num_classes=3)
        f1 = F1Score(task="multiclass", num_classes=3, average='macro')

        valid_accuracy_score = acc(entailment_pred_classes, all_entailment_labels)
        valid_f1_score = f1(entailment_pred_classes, all_entailment_labels)

        # print(f"Pearson Correlation: {valid_pearson_corr:.4f}")
        # print(f"Spearman Correlation: {valid_spearman_corr:.4f}")
        # print(f"Accuracy: {valid_accuracy_score:.4f}")
        # print(f"F1 Score: {valid_f1_score:.4f}")

    valid_pearson_corr_list.append(valid_pearson_corr.detach().numpy())
    valid_spearman_corr_list.append(valid_spearman_corr.detach().numpy())
    valid_accuracy_list.append(valid_accuracy_score.detach().numpy())
    valid_f1_list.append(valid_f1_score.detach().numpy())

    # torch.save(model, f'./saved_models/ep{ep}.ckpt')
    cur_performance = torch.stack((valid_spearman_corr, valid_accuracy_score)).mean()

    if cur_performance >= best_performance:
        torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
        best_performance = cur_performance


workspace = '/workspace/NLP/HW3/figures'

labels = ["NEUTRAL", "ENTAILMENT", "CONTRADICTION"]

cm = confusion_matrix(torch.argmax(train_entailment_preds, dim=1), train_entailment_labels, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'train_confusion_matrix_normalize.png'))

cm = confusion_matrix(torch.argmax(train_entailment_preds, dim=1), train_entailment_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'train_confusion_matrix.png'))

cm = confusion_matrix(entailment_pred_classes, all_entailment_labels, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'val_confusion_matrix_normalize.png'))

cm = confusion_matrix(entailment_pred_classes, all_entailment_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'val_confusion_matrix.png'))


# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot train and valid Pearson correlation
axs[0, 0].plot(train_pearson_corr_list, label='train_pearson_corr')
axs[0, 0].plot(valid_pearson_corr_list, label='valid_pearson_corr')
axs[0, 0].legend()
axs[0, 0].set_title('Pearson Correlation')

# Plot train and valid Spearman correlation
axs[0, 1].plot(train_spearman_corr_list, label='train_spearman_corr')
axs[0, 1].plot(valid_spearman_corr_list, label='valid_spearman_corr')
axs[0, 1].legend()
axs[0, 1].set_title('Spearman Correlation')

# Plot train and valid accuracy
axs[1, 0].plot(train_accuracy_list, label='train_accuracy')
axs[1, 0].plot(valid_accuracy_list, label='valid_accuracy')
axs[1, 0].legend()
axs[1, 0].set_title('Accuracy')

# Plot train and valid F1 score
axs[1, 1].plot(train_f1_list, label='train_f1')
axs[1, 1].plot(valid_f1_list, label='valid_f1')
axs[1, 1].legend()
axs[1, 1].set_title('F1 Score')

# Use tight layout
plt.tight_layout()

# Save the combined plot
plt.savefig(os.path.join(workspace, plot_name))



print([item.item() for item in train_pearson_corr_list])
print([item.item() for item in valid_pearson_corr_list])
print([item.item() for item in train_spearman_corr_list])
print([item.item() for item in valid_spearman_corr_list])
print([item.item() for item in train_accuracy_list])
print([item.item() for item in valid_accuracy_list])
print([item.item() for item in train_f1_list])
print([item.item() for item in valid_f1_list])

"""
For test set predictions, you can write perform evaluation simlar to #TODO5.
"""
model.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name), weights_only=True))
model.eval()


pbar = tqdm(dl_test)
# Create lists to store predictions

all_relatedness_preds = []
all_relatedness_labels = []
all_entailment_preds = []
all_entailment_labels = []
# TODO5: Write the evaluation loop
# Write your code here
# Evaluate your model
# Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)

with torch.no_grad():
    for inputs, relatedness_scores, entailment_judgments in pbar:
        # Move data to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        relatedness_scores = relatedness_scores.to(device)
        entailment_judgments = entailment_judgments.to(device)
        # Forward pass
        relate_pred, entailment_pred = model(**inputs)
        # Collect predictions and labels
        all_relatedness_preds.append(relate_pred.cpu())
        all_relatedness_labels.append(relatedness_scores.cpu())
        all_entailment_preds.append(entailment_pred.cpu())
        all_entailment_labels.append(entailment_judgments.cpu())

    all_relatedness_preds = torch.cat(all_relatedness_preds)
    all_relatedness_labels = torch.cat(all_relatedness_labels)
    all_entailment_preds = torch.cat(all_entailment_preds)
    all_entailment_labels = torch.cat(all_entailment_labels)

    # Compute Spearman correlation
    pearson = PearsonCorrCoef()
    pearson_corr = pearson(all_relatedness_preds, all_relatedness_labels)
    spc = SpearmanCorrCoef()
    spearman_corr = spc(all_relatedness_preds, all_relatedness_labels)

    # Compute accuracy and F1 score
    entailment_pred_classes = torch.argmax(all_entailment_preds, dim=1)
    acc = Accuracy(task="multiclass", num_classes=3)
    f1 = F1Score(task="multiclass", num_classes=3, average='macro')

    accuracy_score = acc(entailment_pred_classes, all_entailment_labels)
    f1_score = f1(entailment_pred_classes, all_entailment_labels)

    print(f"\n")
    print(f"Test Pearson Correlation: {pearson_corr:.4f}")
    print(f"Test Spearman Correlation: {spearman_corr:.4f}")
    print(f"Test Accuracy: {accuracy_score:.4f}")
    print(f"Test F1 Score: {f1_score:.4f}")


cm = confusion_matrix(entailment_pred_classes, all_entailment_labels, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'test_confusion_matrix_normalized.png'))


cm = confusion_matrix(entailment_pred_classes, all_entailment_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(12, 8))
fig.tight_layout()
plt.rcParams.update({'font.size': 16})
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.tick_params(labelsize=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Confusion Matrix")
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.savefig(os.path.join(workspace, 'test_confusion_matrix.png'))
