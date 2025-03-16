import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import os
# Import your configurations
from config import model, device
from time import time
# Import your dataset classes
from dataset import TextDataset, MusicDataset
import wandb


def get_dataloader(data_dir, set_name, type, batch_size=64, max_length=512, sample_size=None):
    if type == 'text':
        dataset = TextDataset(
        data_dir=data_dir,
        set_name=set_name,
        sample_size=sample_size,
        max_length=max_length,
        )
    else:
        dataset = MusicDataset(
        data_dir=data_dir,
        set_name=set_name,
        sample_size=sample_size,
        max_length=max_length
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if set_name == 'train' else False,
        collate_fn=lambda batch: default_collate([b for b in batch if b is not None])
    )

    return loader


def train_one_epoch(train_loader_text, train_loader_music, model, 
                    optimizer, criterion):
    """
    Run one epoch of training over the text and music data in parallel (zipping them).
    """
    model.train()
    total_loss, total_music_loss, total_text_loss = 0.0, 0.0, 0.0
    total_steps = 0
    correct_text, total_text = 0, 0
    correct_music, total_music = 0, 0
    

    for batch_idx, (xt_yt, xm_ym) in enumerate(zip(train_loader_text, train_loader_music)):
        # Unpack text data
        xti, xta, yt = xt_yt[0].to(device), xt_yt[1].to(device), xt_yt[2].to(device)
        xt = {
            'input_ids': xti,
            'attention_mask': xta
        }

        # Unpack music data
        xm, ym = xm_ym[0].to(device), xm_ym[1].to(device)

        # Forward
        logits_text = model(xt, type='text')
        logits_music = model(xm, type='music')

        # Compute loss
        loss_text = criterion(logits_text, yt)
        loss_music = criterion(logits_music, ym)
        loss = loss_text + loss_music
        
        preds_text = torch.argmax(logits_text, dim=1)
        correct_text += (preds_text == yt).sum().item()
        total_text += yt.size(0)
        
        preds_music = torch.argmax(logits_music, dim=1)
        correct_music += (preds_music == ym).sum().item()
        total_music += ym.size(0)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        total_text_loss += loss_text.item()
        total_music_loss += loss_music.item()
        total_steps += 1

        
        if batch_idx % 10 == 0:
            print(f"[Train] Batch {batch_idx}/{len(train_loader_text)} | Loss: {loss.item():.4f}")

        wandb.log(
            {
                "step": total_steps,
                "train":
                    {
                        "train_both_loss": loss.item(),
                        "train_text_loss": loss_text.item(),
                        "train_music_loss": loss_music.item(),
                    }, 
                
            }
        )
        
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    avg_text_loss = total_text_loss / total_steps if total_steps > 0 else 0
    avg_music_loss = total_music_loss / total_steps if total_steps > 0 else 0
    text_acc = correct_text / total_text if total_text > 0 else 0
    music_acc = correct_music / total_music if total_music > 0 else 0

    return avg_loss, avg_text_loss, avg_music_loss, text_acc, music_acc



def evaluate(model, dataloader, criterion, type='music'):
    """
    Run inference on the evaluation set and compute metrics.
    This is a simple example computing accuracy over text & music.
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    total_steps = 0
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            # Unpack text data
            if type == 'text':
                xti, xta, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x = {
                    'input_ids': xti,
                    'attention_mask': xta
                }
            else:
                x, y = batch[0].to(device), batch[1].to(device)

            # Forward
            logits = model(x, type=type)
            loss = criterion(logits, y)
            total_loss += loss.item()
            total_steps += 1
            # Predicted class is the argmax of logits along dim=1
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    # Compute accuracies
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    acc = correct / total if total > 0 else 0
    return avg_loss, acc



import argparse


def parse_args():

    parser = argparse.ArgumentParser(description="Train SharedModel with text and music datasets.")

    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=512, help="Batch size for training")
    parser.add_argument("--sample_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="train_logs/ckpt", help="Checkpoint directory")
    
    return parser.parse_args()

def main():
    # Hyperparameters
    args = parse_args()
    train_config = args.__dict__
    name = ''
    for k, v in train_config.items():
        if k == 'ckpt_dir': continue
        name += f"{k}={v}_"
    name += str(time())
    wandb.init(
        project="music-text-temporal-relation",  # 你的项目名称
        name=name,  # 运行的名称，可以根据时间或实验参数命名
        config=train_config
    )
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    ckpt_dir = os.path.join(args.ckpt_dir, name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    sample_size = args.sample_size
    max_length = args.max_length
    # Prepare Datasets
    train_loader_text = get_dataloader('data/text', 'train', 'text', batch_size, max_length, sample_size)
    train_loader_music = get_dataloader('data/midi_oct', 'train', 'music', batch_size, max_length, sample_size)
    test_loader_text = get_dataloader('data/text', 'test', 'text', batch_size, max_length, sample_size)
    test_loader_music = get_dataloader('data/midi_oct', 'test', 'music', batch_size, max_length, sample_size)
    val_loader_text = get_dataloader('data/text', 'valid', 'text', batch_size, max_length, sample_size)
    val_loader_music = get_dataloader('data/midi_oct', 'valid', 'music', batch_size, max_length, sample_size)
    
    # Move model to device
    model.to(device)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    min_loss = float('inf')
    
    # Training loop
    for epoch in tqdm(range(1, epochs + 1)):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        print("=== Training ===")
        avg_train_loss, avg_text_loss, avg_music_loss, text_acc, music_acc = train_one_epoch(train_loader_text, train_loader_music, model, optimizer, criterion)
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            model.save_weights(os.path.join(ckpt_dir, f"checkpoints_best.pth"))
        
        print(f"[Epoch {epoch}] Average Training Loss: {avg_train_loss:.4f}")
        print(f"[Epoch {epoch}] Average Text Loss: {avg_text_loss:.4f}")
        print(f"[Epoch {epoch}] Text Accuracy: {text_acc * 100:.2f}%")
        print(f"[Epoch {epoch}] Average Music Loss: {avg_music_loss:.4f}")
        print(f"[Epoch {epoch}] Music Accuracy: {music_acc * 100:.2f}%")

        print("=== Validating ===")
        val_text_loss, val_text_acc = evaluate(model, val_loader_text, criterion, 'text')
        val_music_loss, val_music_acc = evaluate(model, val_loader_music, criterion, 'music')
        print(f"[Epoch {epoch}] Val Text Loss: {val_text_loss:.4f}")
        print(f"[Epoch {epoch}] Val Text Accuracy: {val_text_acc * 100:.2f}%")
        print(f"[Epoch {epoch}] Val Music Loss: {val_music_loss:.4f}")
        print(f"[Epoch {epoch}] Val Music Accuracy: {val_music_acc * 100:.2f}%")
        
        print("=== Testing ===")
        test_text_loss, test_text_acc = evaluate(model, test_loader_text, criterion, 'text')
        test_music_loss, test_music_acc = evaluate(model, test_loader_music, criterion, 'music')
        print(f"[Epoch {epoch}] Test Text Loss: {test_text_loss:.4f}")
        print(f"[Epoch {epoch}] Test Text Accuracy: {test_text_acc * 100:.2f}%")
        print(f"[Epoch {epoch}] Test Music Loss: {test_music_loss:.4f}")
        print(f"[Epoch {epoch}] Test Music Accuracy: {test_music_acc * 100:.2f}%")
        
        wandb.log(
        { 
            "epoch": epoch,
            "min_loss": min_loss,
            "best_epoch": epoch,
            "train": {   
                "loss": avg_train_loss,
                "text": {
                    "loss": avg_text_loss,
                    "acc": text_acc,
                    },
                "music": {
                    "loss": avg_music_loss,
                    "acc": music_acc,
                    }
                },
            "val": {
                "text": {
                    "loss": val_text_loss,
                    "acc": val_text_acc,
                    },
                "music": {
                    "loss": val_music_loss,
                    "acc": val_music_acc,
                    }
                },
            "test": {
                "text": {
                    "loss": test_text_loss,
                    "acc": test_text_acc,
                    },
                "music": {
                    "loss": test_music_loss,
                    "acc": test_music_acc,
                    }
                
            }
            }
        )
        
        model.save_weights(os.path.join(ckpt_dir, f"checkpoints_epoch{epoch}.pth"))


if __name__ == "__main__":
    main()