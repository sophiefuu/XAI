from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from util import (AverageMeter, Ham10000, initialize_model, process_metadata, oversample,
                  plot_confusion_matrix, set_seeds, img_stats, store_img_stats)


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    stats = {
        "loss": train_loss.avg,
        "acc": train_acc.avg
    }
    progress = tqdm(
        train_loader, desc=f"Epoch {epoch}", postfix=stats, leave=False)
    for data in progress:
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.size(0), 'label shape',labels.size(0))
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(
            labels.view_as(prediction)).sum().item() / N)
        train_loss.update(loss.item())
        stats = {
            "loss": train_loss.avg,
            "acc": train_acc.avg
        }
        progress.set_postfix(stats)
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, optimizer, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item() / N)

            val_loss.update(criterion(outputs, labels).item())
    return val_loss.avg, val_acc.avg


if __name__ == "__main__":
    set_seeds(0xC0FFEE)

    EPOCHS = 20
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    LEARNING_RATE = 5e-4
    MODEL_NAME = "efficientnet_b3a"
    FEATURE_EXTRACTION = False
    tqdm.write(f"Using device {DEVICE}")

    # Directories where store and retreive data
    this_dir = Path(__file__).parent
    data_dir = this_dir/"data"
    img_paths = list(data_dir.glob("**/*.jpg"))
    result_dir = this_dir/"results"/datetime.now().strftime("%y-%m-%dT%H-%M")
    result_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write("Processing metadata")
    metadata = process_metadata(data_dir/"metadata.csv", img_paths)
    # Prevent duplicates to be split into train and validation
    unduplicated = metadata[["lesion_id", "cell_type_idx"]].drop_duplicates()
    train_ids, val_ids = train_test_split(
        unduplicated["lesion_id"],
        test_size=0.2,
        stratify=unduplicated["cell_type_idx"]
    )
    df_train = metadata[metadata["lesion_id"].isin(train_ids)]
    df_val = metadata[metadata["lesion_id"].isin(val_ids)]

    # build model
    num_classes = metadata["cell_type_idx"].nunique()
    

    # Initialize model
    model, input_size = initialize_model(
        MODEL_NAME, num_classes, FEATURE_EXTRACTION, use_pretrained=True, drop_rate=0.5
    )

    # Define the device:
    
    # Put the model on the device:
    model = model.to(DEVICE)
    # define the transformation of the train images.

    # equalize sampling
    tqdm.write("Computing stats for training set:")
    means, stdevs = img_stats(df_train["path"], input_size)
    store_img_stats(result_dir/"train_img_stats.json", means, stdevs)
    df_train.to_csv(result_dir/"train.csv", index=False)
    df_train = oversample(df_train)

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(means, stdevs),
    ])
    # define the transformation of the val images.
    tqdm.write("Computing stats for validation set:")
    means, stdevs = img_stats(df_val["path"], input_size)
    # Store image statistics
    store_img_stats(result_dir/"val_img_stats.json", means, stdevs)    
    df_val.to_csv(result_dir/"val.csv", index=False)

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(means, stdevs),
    ])

    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = Ham10000(df_train, transform=train_transform)
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # Same for the validation set:
    validation_set = Ham10000(df_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=8)

    # use Adam optimizer, use cross entropy loss as our loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Model training
    
    best_val_acc = 0

    train_stats = pd.DataFrame()
    progress = tqdm(range(EPOCHS), desc="Training", leave=True)
    for epoch in progress:
        loss_train, acc_train = train(
            train_loader, model, criterion, optimizer, epoch, DEVICE)
        loss_val, acc_val = validate(
            val_loader, model, criterion, optimizer, epoch, DEVICE)

        train_stats = train_stats.append({
            "epoch": epoch,
            "train loss": loss_train,
            "train acc": acc_train,
            "val loss": loss_val,
            "val acc": acc_val
        }, ignore_index=True)
        progress.set_postfix_str(f"val loss={loss_val:1.03f}, val acc={acc_val:1.03f}")
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), result_dir /
                       f"{MODEL_NAME}_weights_{epoch}.pt")

    train_stats.to_csv(result_dir/"train_stats.csv", index=False)
    torch.save(model, result_dir/f"{MODEL_NAME}.pt")
    torch.save(model.state_dict(), result_dir/f"{MODEL_NAME}_weights.pt")

    model.eval()

    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc="Validating")):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(DEVICE)
            outputs = model(images).to(DEVICE)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    np.save(result_dir/"confusion.npy", confusion_mtx)
    # plot the confusion matrix
    plot_labels = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
    plot_confusion_matrix(confusion_mtx, plot_labels,
                          save_path=result_dir/"confusion.pdf")

    # Generate a classification report
    report = pd.DataFrame(
        classification_report(
            y_label, y_predict, target_names=plot_labels, output_dict=True
        ))
    report.to_csv(result_dir/"report.csv")
    print(report)
