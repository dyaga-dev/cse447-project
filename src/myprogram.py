#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import torch.nn.functional as F
import os
import torch.nn as nn
from collections import Counter

# LSTM model suggested by Claude, reference: https://medium.com/data-science/language-modeling-with-lstms-in-pytorch-381a26badcbf
class CharacterLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        emb     = self.dropout(self.embedding(x))  
        out, _  = self.lstm(emb)                    
        logits  = self.fc(out[:, -1, :])             
        return logits

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    
    @classmethod
    def load_training_data(cls, dpath):
        # your code here
        # this particular model doesn't train
        try:
            data = []
            if os.path.isdir(dpath):
                for file in sorted(os.listdir(dpath)):
                    path = os.path.join(dpath, file)
                    with open(path, encoding = "utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(line)
            else:
                with open(dpath, encoding="utf-8") as f:
                    for line in f:
                        inp = line.strip()  # the last character is a newline
                        data.append(inp)
            random.shuffle(data)
            return data
        except Exception as e:
            print("error in load_training_data: " + e)

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        try: 
            data = []
            with open(fname) as f:
                for line in f:
                    inp = line.strip()  # the last character is a newline
                    data.append(inp)
            return data
        except Exception as e:
            print("error in load_test_data: " + e)

    @classmethod
    def write_pred(cls, preds, fname):
        try: 
            with open(fname, 'wt') as f:
                for p in preds:
                    f.write('{}\n'.format(p))
        except Exception as e:
            print("error in write_pred: " + e)

    def run_train(self, data, work_dir):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MIN_COUNT = 5 # suggested by claude to reduce softmax operations 
            all_text = "\n".join(data)
            char_counts = Counter(all_text)
            keep = sorted(c for c, n in char_counts.items() if n >= MIN_COUNT)
            if "." in keep:
                keep.remove(".")
        
            chars = ['.'] + keep          # start token
            ctoi = {ch:i for i,ch in enumerate(chars)}
            itoc = {i:ch for ch,i in ctoi.items()}
            vocab_size = len(chars)
            print(ctoi)
            block_size = 20

            x, y = [], []
            PAD = ctoi["."]

            for sentence in data:
                encoded = [ctoi[ch] for ch in sentence if ch in ctoi]
                if len(encoded) < 2:
                    continue

                context = [PAD] * block_size + encoded
                for i in range(block_size, len(context)):
                    x.append(context[i - block_size : i])
                    y.append(context[i])
                    
            X = torch.tensor(x, dtype = torch.long)
            Y = torch.tensor(y, dtype = torch.long)

            embed_dim = 128
            hidden_size = 512
            num_layers = 2
            dropout = 0.3
            model = CharacterLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Halve LR every 10 000 steps suggested by claude
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.5)

            batch_size = 256
            steps = 5000
            n = X.shape[0]
            model.train()
            print(f"\n{'─'*68}")
            print(f"  Steps: {steps:,} | Batch: {batch_size} | "
                  f"Vocab: {vocab_size:,} | Examples: {n:,}")
            print(f"{'─'*68}\n")
            for step in range(steps):
                idx  = torch.randint(0, n, (batch_size,))
                xb   = X[idx].to(device)
                yb   = Y[idx].to(device)

                logits = model(xb)
                loss   = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if step % 250 == 0:
                    print(f"step {step} | loss {loss.item():.4f}")
            os.makedirs(work_dir, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "ctoi":        ctoi,
                "itoc":        itoc,
                "block_size":  block_size,
                "embed_dim":   embed_dim,
                "hidden_size": hidden_size,
                "num_layers":  num_layers,
                "dropout":     dropout,
                "vocab_size":  vocab_size,
            }, os.path.join(work_dir, "model.pt"))
            print("Model saved to", work_dir)

        except Exception as e:
            print("error in run_test: " + e)
        

    def run_pred(self, data, work_dir):
        # your code here
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            checkpoint = torch.load(os.path.join(work_dir, "model.pt"), map_location =device)

            ctoi       = checkpoint["ctoi"]
            itoc       = checkpoint["itoc"]
            block_size = checkpoint["block_size"]
            vocab_size = checkpoint["vocab_size"]

            model = CharacterLSTM(
                vocab_size  = vocab_size,
                embed_dim   = checkpoint["embed_dim"],
                hidden_size = checkpoint["hidden_size"],
                num_layers  = checkpoint["num_layers"],
                dropout     = checkpoint["dropout"],
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            PAD = ctoi["."]
            batch_size = 128

            final_preds = [""]*len(data)

            # preprocess data first
            contexts = []
            for line in data:
                s = line.strip()
                if not s:
                    contexts.append(None)
                    continue
                ctx = [PAD] * block_size
                for ch in s[-block_size:]:
                    if ch in ctoi:
                        ctx = ctx[1:] + [ctoi[ch]]
                    else:
                        ctx = ctx[1:] + [PAD]
                contexts.append(ctx)
            indices = [i for i, c in enumerate(contexts) if c is not None]

            
            for batch_start in range(0, len(indices), batch_size):
                batch_idx = indices[batch_start : batch_start + batch_size]
                x_b = torch.tensor([contexts[i] for i in batch_idx], dtype=torch.long).to(device)

                with torch.no_grad():
                    logits = model(x_b)                     
                    probs  = F.softmax(logits, dim=-1)          

                top_probs, top_idx = torch.topk(probs, 3, dim=-1) 

                for j, orig_i in enumerate(batch_idx):
                    chars = "".join(itoc[idx.item()] for idx in top_idx[j])
                    final_preds[orig_i] = chars

            return final_preds
        except Exception as e:
            print("error in run_test: " + e)

    def save(self, work_dir):
        try:
            os.makedirs(work_dir, exist_ok=True)
            path = os.path.join(work_dir, "model.checkpoint")
            with open(path, "w") as f:
                if hasattr(self, "args"):
                    f.write(f"{self.args.mode}\n")
                    f.write(f"{self.args.work_dir}\n")
                    f.write(f"{self.args.test_data}\n")
                    f.write(f"{self.args.test_output}\n")
                else:
                    f.write("no-args\n")
        except Exception as e:
            print("error in save: " + e)

    @classmethod
    def load(cls, work_dir):
        try:
            model = cls()
            path = os.path.join(work_dir, "model.checkpoint")
            with open(path, "r") as f:
                lines = [line.rstrip("\n") for line in f.readlines()]

            # Only restore args if you actually want them
            # (and if your model has/needs an args object).
            return model
        except Exception as e:
            print("error in load: " + e)
            return None



if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    train_data_dir = 'multilingual_dataset'
    test_data_file = 'testing_data/all_test_input.txt'

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data(train_data_dir)
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        if args.test_data == 'example/input.txt':
            args.test_data = test_data_file

        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data, args.work_dir)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
