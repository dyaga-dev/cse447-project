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
import itertools
import copy

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
    def _build_vocab_and_xy(self, data, block_size, ctoi=None):
        print(f"Building vocab/X/Y for {len(data)} sentences (block_size={block_size})", flush=True)

        all_text = "\n".join(data)

        if ctoi is None:
            MIN_COUNT = 5
            char_counts = Counter(all_text)
            keep = sorted(c for c, n in char_counts.items() if n >= MIN_COUNT)
            if "." in keep:
                keep.remove(".")
            chars = ['.'] + keep
            ctoi = {ch: i for i, ch in enumerate(chars)}
            itoc = {i: ch for ch, i in ctoi.items()}
        else:
            itoc = {i: ch for ch, i in ctoi.items()}

        print(f"Vocabulary size: {len(ctoi)}", flush=True)
        print("Generating training examples...", flush=True)

        x, y = [], []
        PAD = ctoi["."]

        for idx, sentence in enumerate(data):
            if idx % 50000 == 0:
                print(f"Processed {idx}/{len(data)} sentences...", flush=True)
            encoded = [ctoi[ch] for ch in sentence if ch in ctoi]
            if len(encoded) < 2:
                continue

            context = [PAD] * block_size + encoded
            for i in range(block_size, len(context)):
                x.append(context[i - block_size:i])
                y.append(context[i])

        if len(x) == 0:
            return ctoi, itoc, None, None

        print(f"Created {len(x)} examples. Converting to tensors...", flush=True)
        X = torch.tensor(x, dtype=torch.long)
        Y = torch.tensor(y, dtype=torch.long)
        print("Tensor conversion complete.", flush=True)

        return ctoi, itoc, X, Y
    def _build_or_load_xy(self, data, block_size, cache_dir, cache_name, ctoi=None):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_name}_bls{block_size}.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached tensors from {cache_path}", flush=True)
            obj = torch.load(cache_path)
            return obj["ctoi"], obj["itoc"], obj["X"], obj["Y"]

        print(f"Cache miss for {cache_path}", flush=True)
        ctoi2, itoc2, X, Y = self._build_vocab_and_xy(data, block_size, ctoi=ctoi)

        torch.save({
            "ctoi": ctoi2,
            "itoc": itoc2,
            "X": X,
            "Y": Y,
        }, cache_path)

        print(f"Saved cache to {cache_path}", flush=True)
        return ctoi2, itoc2, X, Y

    def _eval_top3_accuracy(self, model, X, Y, batch_size=256, device=None):
        if X is None or Y is None or len(X) == 0:
            return 0.0

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = X[start:start + batch_size].to(device)
                yb = Y[start:start + batch_size].to(device)
                logits = model(xb)
                top3 = torch.topk(logits, 3, dim=-1).indices
                correct += (top3 == yb.unsqueeze(1)).any(dim=1).sum().item()
                total += yb.size(0)

        return correct / total if total > 0 else 0.0
    
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

    def run_train(
        self,
        data,
        work_dir,
        bls=64,
        ed=64,
        hs=512,
        nl=2,
        do=0.3,
        bas=256,
        st=5000,
        lr=1e-3,
        val_data=None,
    ):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            block_size = bls
            cache_dir = os.path.join(work_dir, "cache")

            print("Building/loading training tensors...", flush=True)
            ctoi, itoc, X, Y = self._build_or_load_xy(
                data, block_size, cache_dir, "train", ctoi=None
            )
            if X is None or Y is None:
                print("No training examples were created.")
                return 0.0

            vocab_size = len(ctoi)
            # print(ctoi)

            embed_dim = ed
            hidden_size = hs
            num_layers = nl
            dropout = do

            model = CharacterLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.5)

            batch_size = bas
            steps = st
            n = X.shape[0]

            val_X, val_Y = None, None
            if val_data is not None:
                print("Building/loading validation tensors...", flush=True)
                _, _, val_X, val_Y = self._build_or_load_xy(
                    val_data, block_size, cache_dir, "val", ctoi=ctoi
                )
            model.train()
            print(f"\n{'─'*68}")
            print(f"  Steps: {steps:,} | Batch: {batch_size} | Vocab: {vocab_size:,} | Examples: {n:,}")
            print(f"{'─'*68}\n")

            best_val = -1.0
            best_state = None

            for step in range(steps):
                idx = torch.randint(0, n, (batch_size,))
                xb = X[idx].to(device)
                yb = Y[idx].to(device)

                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if step == 0:
                    print(f"step 0 | loss {loss.item():.4f}", flush=True)
                elif step % 250 == 0:
                    msg = f"step {step} | loss {loss.item():.4f}"
                    if val_X is not None:
                        val_acc = self._eval_top3_accuracy(model, val_X, val_Y, device=device)
                        msg += f" | val_top3 {val_acc:.4f}"
                        if val_acc > best_val:
                            best_val = val_acc
                            best_state = copy.deepcopy(model.state_dict())
                    print(msg)

            if best_state is not None:
                model.load_state_dict(best_state)

            os.makedirs(work_dir, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "ctoi": ctoi,
                "itoc": itoc,
                "block_size": block_size,
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "vocab_size": vocab_size,
            }, os.path.join(work_dir, "model.pt"))
            print("Model saved to", work_dir)

            if val_X is not None:
                final_val = self._eval_top3_accuracy(model, val_X, val_Y, device=device)
                print(f"final val_top3 = {final_val:.4f}")
                return final_val

            return 0.0

        except Exception as e:
            print("error in run_train:", e)
            return 0.0
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
        # print('Training')
        # model.run_train(train_data, args.work_dir)
        print(f"Loaded {len(train_data)} total training lines", flush=True)

        TUNE_LIMIT = 200000
        tune_data = train_data[:TUNE_LIMIT]

        print(f"Using {len(tune_data)} lines for hyperparameter tuning", flush=True)
        print('Preparing train/validation split for tuning', flush=True)
        split = int(0.9 * len(tune_data))
        train_split = tune_data[:split]
        val_split = tune_data[split:]

        search_space = [
            {"bls": 32, "ed": 64,  "hs": 256, "nl": 1, "do": 0.2, "bas": 256, "st": 3000, "lr": 1e-3},
            {"bls": 64, "ed": 64,  "hs": 512, "nl": 2, "do": 0.3, "bas": 256, "st": 3000, "lr": 1e-3},
            {"bls": 64, "ed": 128, "hs": 512, "nl": 2, "do": 0.3, "bas": 256, "st": 5000, "lr": 1e-3},
            {"bls": 96, "ed": 128, "hs": 512, "nl": 2, "do": 0.4, "bas": 128, "st": 5000, "lr": 1e-3},
            {"bls": 64, "ed": 64,  "hs": 512, "nl": 2, "do": 0.3, "bas": 256, "st": 5000, "lr": 3e-4},
            {"bls": 64, "ed": 128, "hs": 768, "nl": 2, "do": 0.3, "bas": 128, "st": 5000, "lr": 3e-4},
        ]

        best_cfg = None
        best_score = -1.0

        for i, cfg in enumerate(search_space):
            trial_dir = os.path.join(args.work_dir, f"trial_{i}")
            print(f"\n=== Trial {i} / {len(search_space)} ===")
            print(cfg)
            trial_model = MyModel()
            score = trial_model.run_train(train_split, trial_dir, val_data=val_split, **cfg)
            print(f"trial {i} val_top3 = {score:.4f}")

            if score > best_score:
                best_score = score
                best_cfg = cfg

        print("\nBest config:", best_cfg)
        print(f"Best validation top-3 accuracy: {best_score:.4f}")

        print('\nRetraining best config on full training data')
        model = MyModel()
        model.run_train(train_data, args.work_dir, **best_cfg)
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
