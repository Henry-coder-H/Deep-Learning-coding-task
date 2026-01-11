import argparse
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.utils import CRNN_ALPHABET
from recognition.datasets import LPRDataset
from recognition.model import CRNN
from recognition.utils import CTCLabelConverter, save_checkpoint


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset & Dataloader
    train_set = LPRDataset(args.train_labels, img_w=args.imgw, img_h=args.imgh)
    val_set = LPRDataset(args.val_labels, img_w=args.imgw, img_h=args.imgh)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    
    print(f"[INFO] Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    # 2. Model
    converter = CTCLabelConverter(CRNN_ALPHABET)
    num_classes = len(CRNN_ALPHABET) + 1  # +1 for blank
    
    model = CRNN(num_classes=num_classes, img_h=args.imgh, nc=1)
    model.to(device)
    
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        
    criterion = nn.CTCLoss(blank=converter.blank, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    save_dir = Path("runs/rec")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, texts in pbar:
            imgs = imgs.to(device)
            targets, target_lengths = converter.encode(texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            preds = model(imgs) # [T, B, C]
            
            # preds size: [T, B, C]
            # log_softmax for CTCLoss
            preds_log_softmax = preds.log_softmax(2)
            
            input_lengths = torch.full(
                (imgs.size(0),), preds.size(0), dtype=torch.long, device=device
            )
            
            loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
            
            # DEBUG: Print first batch details once to check sanity
            if epoch == 0 and total_loss == 0: # First iteration
                print(f"\n[DEBUG] Input Img Range: {imgs.min():.2f} ~ {imgs.max():.2f}")
                print(f"[DEBUG] Pred Shape: {preds.shape}") # [T, B, C]
                # Try decoding first sample in batch
                t = preds[:, 0, :].argmax(1) # [T]
                print(f"[DEBUG] Raw Argmax (First Sample): {t.tolist()}")
                
                # Manual decode to check
                chars = []
                prev = converter.blank
                for s in t.tolist():
                    if s != prev and s != converter.blank:
                        chars.append(converter.alphabet[s])
                    prev = s
                print(f"[DEBUG] Decoded (First Sample): {''.join(chars)}")
                print(f"[DEBUG] GT (First Sample): {texts[0]}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, texts in tqdm(val_loader, desc="Val"):
                imgs = imgs.to(device)
                preds = model(imgs) # [T, B, C]
                
                # Decode
                T, B, _ = preds.shape
                # Greedy decode output needs softmax to choose best path? 
                # CTCLabelConverter.decode usually takes raw logits or log_softmax
                # But let's check decode implementation.
                # Usually argmax on logits is fine.
                
                pred_lens = torch.full((B,), T, dtype=torch.long, device=device)
                
                decoded_texts = converter.decode(preds, pred_lens)
                
                for pred_text, gt_text in zip(decoded_texts, texts):
                    if pred_text == gt_text:
                        correct += 1
                    total += 1
                    
            # Debug print (first batch only usually, but here we print last batch sample)
            if total > 0 and len(decoded_texts) > 0:
                print(f"\n[DEBUG] Sample Pred: {decoded_texts[0]} | GT: {texts[0]}")
        
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} Acc: {acc:.4f}")
        
        # Save best
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model.state_dict(), save_dir / "best.pth")
            print(f"[INFO] Saved best model (acc={best_acc:.4f})")
            
        save_checkpoint(model.state_dict(), save_dir / "last.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--val-labels", type=str, required=True)
    parser.add_argument("--imgw", type=int, default=160)
    parser.add_argument("--imgh", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()



