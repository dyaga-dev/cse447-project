#!/usr/bin/env python
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter, defaultdict

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
    def build_vocab(self, data, min_count):
        char_counts = Counter()
        for inp, ans in data:
            char_counts.update(inp)
            char_counts.update(ans)

        keep = sorted(c for c, n in char_counts.items() if n >= min_count)
        if "." in keep:
            keep.remove(".")
        chars = ['.'] + keep
        ctoi = {ch:i for i,ch in enumerate(chars)}
        itoc = {i:ch for ch,i in ctoi.items()}
        return ctoi, itoc

    def build_xy(self, data, ctoi, block_size):
        x, y = [], []
        PAD = ctoi["."]

        for pair_i, (inp, ans) in enumerate(data):
            if pair_i % 50000 == 0:
                print(f"Processed {pair_i}/{len(data)} pairs...")

            if not ans:
                continue

            target_char = ans[0]
            if target_char not in ctoi:
                continue

            ctx = [PAD] * block_size
            for ch in inp[-block_size:]:
                if ch in ctoi:
                    ctx = ctx[1:] + [ctoi[ch]]
                else:
                    ctx = ctx[1:] + [PAD]

            x.append(ctx)
            y.append(ctoi[target_char])

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def build_ngram_counts(self, data, max_order, ctoi):
        print(f"Building n-gram counts up to order {max_order}...")
        ngram_counts = [defaultdict(Counter) for _ in range(max_order)]

        for pair_i, (inp, ans) in enumerate(data):
            if pair_i % 50000 == 0:
                print(f"Processed {pair_i}/{len(data)} pairs for n-grams...")

            if not ans:
                continue

            target_char = ans[0]
            if target_char not in ctoi:
                continue

            target_idx = ctoi[target_char]

            for order in range(1, max_order + 1):
                suffix = inp[-order:] if len(inp) >= order else inp
                ngram_counts[order - 1][suffix][target_idx] += 1

        # convert nested defaultdict/Counter to normal dict for safe torch.save
        serializable = []
        for order_dict in ngram_counts:
            serializable_order = {}
            for suffix, ctr in order_dict.items():
                serializable_order[suffix] = dict(ctr)
            serializable.append(serializable_order)

        return serializable

    def prepare_ngram_counts(self, data, ctoi, cache_path, max_order):
        if os.path.exists(cache_path):
            print(f"Loading cached n-gram counts from {cache_path}")
            cache = torch.load(cache_path)
            return cache["ngram_counts"]

        print(f"Cache miss for {cache_path}")
        ngram_counts = self.build_ngram_counts(data, max_order, ctoi)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "ngram_counts": ngram_counts,
        }, cache_path)
        print(f"Saved n-gram cache to {cache_path}")
        return ngram_counts

    def get_ngram_probs(self, s, ngram_counts, ctoi, vocab_size, max_order):
        probs = torch.zeros(vocab_size, dtype=torch.float32)

        for order in range(max_order, 0, -1):
            suffix = s[-order:] if len(s) >= order else s
            order_dict = ngram_counts[order - 1]

            if suffix in order_dict:
                counts = order_dict[suffix]
                total = sum(counts.values())
                if total > 0:
                    for idx, cnt in counts.items():
                        probs[idx] = cnt / total
                    return probs

        return probs

    def eval_loss(self, model, X, Y, device, batch_size=512):
        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = X[start:start+batch_size].to(device)
                yb = Y[start:start+batch_size].to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, reduction="sum")
                total_loss += loss.item()
                total_n += len(xb)
        model.train()
        return total_loss / total_n
    
    def prepare_vocab(self, data, cache_path, min_count):
        if os.path.exists(cache_path):
            print(f"Loading cached vocab from {cache_path}")
            cache = torch.load(cache_path)
            return cache["ctoi"], cache["itoc"]

        print(f"Cache miss for {cache_path}")
        ctoi, itoc = self.build_vocab(data, min_count)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "ctoi": ctoi,
            "itoc": itoc,
        }, cache_path)
        print(f"Saved vocab cache to {cache_path}")
        return ctoi, itoc

    def prepare_xy_only(self, data, ctoi, cache_path, block_size):
        if os.path.exists(cache_path):
            print(f"Loading cached tensors from {cache_path}")
            cache = torch.load(cache_path)
            return cache["X"], cache["Y"]

        print(f"Cache miss for {cache_path}")
        X, Y = self.build_xy(data, ctoi, block_size)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "X": X,
            "Y": Y,
        }, cache_path)
        print(f"Saved cache to {cache_path}")
        return X, Y
    
    # @classmethod
    # def load_training_data(cls, dpath):
    #     # your code here
    #     # this particular model doesn't train
    #     try:
    #         data = []
    #         if os.path.isdir(dpath):
    #             for file in sorted(os.listdir(dpath)):
    #                 path = os.path.join(dpath, file)
    #                 with open(path, encoding = "utf-8") as f:
    #                     for line in f:
    #                         line = line.strip()
    #                         if line:
    #                             data.append(line)
    #         else:
    #             with open(dpath, encoding="utf-8") as f:
    #                 for line in f:
    #                     inp = line.strip()  # the last character is a newline
    #                     data.append(inp)
    #         random.shuffle(data)
    #         return data
    #     except Exception as e:
    #         print(f"error in load_training_data: {e}")
    @classmethod
    def load_training_data(cls, dpath):
        try:
            data = []

            if os.path.isdir(dpath):
                input_files = sorted(
                    f for f in os.listdir(dpath)
                    if f.startswith("train_input_") and f.endswith(".txt")
                )

                for input_file in input_files:
                    suffix = input_file[len("train_input_"):]
                    answer_file = f"train_answer_{suffix}"

                    input_path = os.path.join(dpath, input_file)
                    answer_path = os.path.join(dpath, answer_file)

                    if not os.path.exists(answer_path):
                        print(f"warning: missing matching answer file for {input_file}")
                        continue

                    print(f"Loading paired data from {input_file} and {answer_file}")

                    with open(input_path, encoding="utf-8") as fin, open(answer_path, encoding="utf-8") as fans:
                        for inp, ans in zip(fin, fans):
                            inp = inp.rstrip("\n")
                            ans = ans.rstrip("\n")

                            if not inp or not ans:
                                continue

                            data.append((inp, ans))

            else:
                raise ValueError(f"{dpath} is not a directory of paired training files")

            random.shuffle(data)
            print(f"Loaded {len(data)} paired training examples")
            return data

        except Exception as e:
            print(f"error in load_training_data: {e}")


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
            print(f"error in load_test_data: {e}")

    @classmethod
    def write_pred(cls, preds, fname):
        try: 
            with open(fname, 'wt') as f:
                for p in preds:
                    f.write('{}\n'.format(p))
        except Exception as e:
            print(f"error in write_pred: {e}")

    # def run_train(self, data, work_dir, block_size_p = 20, embed_dim_p = 64, hidden_size_p = 512, num_layers_p = 2, dropout_p = 0.3, batch_size_p = 256, steps_p = 5000):
    #     try:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         MIN_COUNT = 5 # suggested by claude to reduce softmax operations 
    #         all_text = "\n".join(data)
    #         char_counts = Counter(all_text)
    #         keep = sorted(c for c, n in char_counts.items() if n >= MIN_COUNT)
    #         if "." in keep:
    #             keep.remove(".")
        
    #         chars = ['.'] + keep          # start token
    #         ctoi = {ch:i for i,ch in enumerate(chars)}
    #         itoc = {i:ch for ch,i in ctoi.items()}
    #         vocab_size = len(chars)
    #         print(ctoi)
    #         block_size = block_size_p

    #         x, y = [], []
    #         PAD = ctoi["."]

    #         for sentence in data:
    #             encoded = [ctoi[ch] for ch in sentence if ch in ctoi]
    #             if len(encoded) < 2:
    #                 continue

    #             context = [PAD] * block_size + encoded
    #             for i in range(block_size, len(context)):
    #                 x.append(context[i - block_size : i])
    #                 y.append(context[i])
                    
    #         X = torch.tensor(x, dtype = torch.long)
    #         Y = torch.tensor(y, dtype = torch.long)

    #         embed_dim = embed_dim_p
    #         hidden_size = hidden_size_p
    #         num_layers = num_layers_p
    #         dropout = dropout_p
    #         model = CharacterLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
    #         model.to(device)

    #         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #         # Halve LR every 10 000 steps suggested by claude
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.5)

    #         batch_size = batch_size_p
    #         steps = steps_p
    #         n = X.shape[0]
    #         model.train()
    #         print(f"\n{'─'*68}")
    #         print(f"  Steps: {steps:,} | Batch: {batch_size} | "
    #               f"Vocab: {vocab_size:,} | Examples: {n:,}")
    #         print(f"{'─'*68}\n")
    #         for step in range(steps):
    #             idx  = torch.randint(0, n, (batch_size,))
    #             xb   = X[idx].to(device)
    #             yb   = Y[idx].to(device)

    #             logits = model(xb)
    #             loss   = F.cross_entropy(logits, yb)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
    #             scheduler.step()
    #             if step % 250 == 0:
    #                 print(f"step {step} | loss {loss.item():.4f}")
    #         os.makedirs(work_dir, exist_ok=True)
    #         torch.save({
    #             "model_state": model.state_dict(),
    #             "ctoi":        ctoi,
    #             "itoc":        itoc,
    #             "block_size":  block_size,
    #             "embed_dim":   embed_dim,
    #             "hidden_size": hidden_size,
    #             "num_layers":  num_layers,
    #             "dropout":     dropout,
    #             "vocab_size":  vocab_size,
    #         }, os.path.join(work_dir, "model.pt"))
    #         print("Model saved to", work_dir)

    #     except Exception as e:
    #         print(f"error in run_test: {e}")
    # def run_train(
    #     self,
    #     data,
    #     work_dir,
    #     block_size_p=20,
    #     embed_dim_p=64,
    #     hidden_size_p=512,
    #     num_layers_p=2,
    #     dropout_p=0.3,
    #     batch_size_p=256,
    #     steps_p=5000,
    #     min_count_p=5,
    #     val_data=None,
    #     cache_dir=None
    # ):
    #     try:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #         block_size = block_size_p
    #         embed_dim = embed_dim_p
    #         hidden_size = hidden_size_p
    #         num_layers = num_layers_p
    #         dropout = dropout_p
    #         batch_size = batch_size_p
    #         steps = steps_p
    #         min_count = min_count_p
    #         eval_every = 500

    #         if cache_dir is None:
    #             cache_dir = os.path.join(work_dir, "cache")
    #         os.makedirs(cache_dir, exist_ok=True)
    #         os.makedirs(work_dir, exist_ok=True)

    #         vocab_cache = os.path.join(cache_dir, f"vocab_mc{min_count}.pt")
    #         ctoi, itoc = self.prepare_vocab(data, vocab_cache, min_count)
    #         vocab_size = len(ctoi)

    #         train_cache = os.path.join(cache_dir, f"train_bs{block_size}_mc{min_count}.pt")
    #         X, Y = self.prepare_xy_only(data, ctoi, train_cache, block_size)
    #         n = X.shape[0]

    #         if n == 0:
    #             raise ValueError("No training examples were created.")

    #         X_val, Y_val = None, None
    #         if val_data is not None and len(val_data) > 0:
    #             val_cache = os.path.join(cache_dir, f"val_bs{block_size}_mc{min_count}.pt")
    #             X_val, Y_val = self.prepare_xy_only(val_data, ctoi, val_cache, block_size)
    #             if len(X_val) == 0:
    #                 X_val, Y_val = None, None

    #         model = CharacterLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
    #         model.to(device)

    #         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    #         model.train()

    #         print(f"\n{'─'*68}")
    #         print(
    #             f"  Steps: {steps:,} | Batch: {batch_size} | "
    #             f"Vocab: {vocab_size:,} | Train examples: {n:,}"
    #         )
    #         if X_val is not None:
    #             print(f"  Val examples: {len(X_val):,} | Min count: {min_count}")
    #         print(f"  Cache dir: {cache_dir}")
    #         print(f"{'─'*68}\n")

    #         best_val_loss = float("inf")

    #         for step in range(steps):
    #             idx = torch.randint(0, n, (batch_size,))
    #             xb = X[idx].to(device)
    #             yb = Y[idx].to(device)

    #             logits = model(xb)
    #             loss = F.cross_entropy(logits, yb)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
    #             scheduler.step()

    #             if step % eval_every == 0:
    #                 if X_val is not None:
    #                     val_loss = self.eval_loss(model, X_val, Y_val, device)
    #                     print(f"step {step} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")

    #                     if val_loss < best_val_loss:
    #                         best_val_loss = val_loss
    #                         torch.save({
    #                             "model_state": model.state_dict(),
    #                             "ctoi": ctoi,
    #                             "itoc": itoc,
    #                             "block_size": block_size,
    #                             "embed_dim": embed_dim,
    #                             "hidden_size": hidden_size,
    #                             "num_layers": num_layers,
    #                             "dropout": dropout,
    #                             "vocab_size": vocab_size,
    #                         }, os.path.join(work_dir, "model.pt"))
    #                         print(f"  saved new best model to {work_dir}")
    #                 else:
    #                     print(f"step {step} | loss {loss.item():.4f}")

    #         if X_val is None:
    #             torch.save({
    #                 "model_state": model.state_dict(),
    #                 "ctoi": ctoi,
    #                 "itoc": itoc,
    #                 "block_size": block_size,
    #                 "embed_dim": embed_dim,
    #                 "hidden_size": hidden_size,
    #                 "num_layers": num_layers,
    #                 "dropout": dropout,
    #                 "vocab_size": vocab_size,
    #             }, os.path.join(work_dir, "model.pt"))
    #             print("Model saved to", work_dir)
    #             return None
    #         else:
    #             print(f"Best val loss: {best_val_loss:.4f}")
    #             return best_val_loss

    #     except Exception as e:
    #         print(f"error in run_train: {e}")
    #         return None 
    def run_train(
        self,
        data,
        work_dir,
        block_size_p=20,
        embed_dim_p=64,
        hidden_size_p=512,
        num_layers_p=2,
        dropout_p=0.3,
        batch_size_p=256,
        steps_p=5000,
        min_count_p=5,
        val_data=None,
        cache_dir=None
    ):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            block_size = block_size_p
            embed_dim = embed_dim_p
            hidden_size = hidden_size_p
            num_layers = num_layers_p
            dropout = dropout_p
            batch_size = batch_size_p
            steps = steps_p
            min_count = min_count_p
            eval_every = 500

            ngram_order = 6
            ngram_alpha = 0.6

            if cache_dir is None:
                cache_dir = os.path.join(work_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)

            vocab_cache = os.path.join(cache_dir, f"vocab_mc{min_count}.pt")
            ctoi, itoc = self.prepare_vocab(data, vocab_cache, min_count)
            vocab_size = len(ctoi)

            train_cache = os.path.join(cache_dir, f"train_bs{block_size}_mc{min_count}.pt")
            X, Y = self.prepare_xy_only(data, ctoi, train_cache, block_size)
            n = X.shape[0]

            if n == 0:
                raise ValueError("No training examples were created.")

            X_val, Y_val = None, None
            if val_data is not None and len(val_data) > 0:
                val_cache = os.path.join(cache_dir, f"val_bs{block_size}_mc{min_count}.pt")
                X_val, Y_val = self.prepare_xy_only(val_data, ctoi, val_cache, block_size)
                if len(X_val) == 0:
                    X_val, Y_val = None, None

            ngram_cache = os.path.join(cache_dir, f"ngram_mc{min_count}_ord{ngram_order}.pt")
            ngram_counts = self.prepare_ngram_counts(data, ctoi, ngram_cache, ngram_order)

            model = CharacterLSTM(vocab_size, embed_dim, hidden_size, num_layers, dropout)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

            model.train()

            print(f"\n{'─'*68}")
            print(
                f"  Steps: {steps:,} | Batch: {batch_size} | "
                f"Vocab: {vocab_size:,} | Train examples: {n:,}"
            )
            if X_val is not None:
                print(f"  Val examples: {len(X_val):,} | Min count: {min_count}")
            print(f"  Cache dir: {cache_dir}")
            print(f"  N-gram order: {ngram_order} | Interp alpha: {ngram_alpha}")
            print(f"{'─'*68}\n")

            best_val_loss = float("inf")

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

                if step % eval_every == 0:
                    if X_val is not None:
                        val_loss = self.eval_loss(model, X_val, Y_val, device)
                        print(f"step {step} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
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
                                "ngram_counts": ngram_counts,
                                "ngram_order": ngram_order,
                                "ngram_alpha": ngram_alpha,
                            }, os.path.join(work_dir, "model.pt"))
                            print(f"  saved new best model to {work_dir}")
                    else:
                        print(f"step {step} | loss {loss.item():.4f}")

            if X_val is None:
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
                    "ngram_counts": ngram_counts,
                    "ngram_order": ngram_order,
                    "ngram_alpha": ngram_alpha,
                }, os.path.join(work_dir, "model.pt"))
                print("Model saved to", work_dir)
                return None
            else:
                print(f"Best val loss: {best_val_loss:.4f}")
                return best_val_loss

        except Exception as e:
            print(f"error in run_train: {e}")
            return None

    # def run_pred(self, data, work_dir):
    #     # your code here
    #     try:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #         checkpoint = torch.load(os.path.join(work_dir, "model.pt"), map_location =device)

    #         ctoi       = checkpoint["ctoi"]
    #         itoc       = checkpoint["itoc"]
    #         block_size = checkpoint["block_size"]
    #         vocab_size = checkpoint["vocab_size"]

    #         model = CharacterLSTM(
    #             vocab_size  = vocab_size,
    #             embed_dim   = checkpoint["embed_dim"],
    #             hidden_size = checkpoint["hidden_size"],
    #             num_layers  = checkpoint["num_layers"],
    #             dropout     = checkpoint["dropout"],
    #         )
    #         model.load_state_dict(checkpoint["model_state"])
    #         model.to(device)
    #         model.eval()
    #         PAD = ctoi["."]
    #         batch_size = 128

    #         final_preds = [""]*len(data)

    #         # preprocess data first
    #         contexts = []
    #         for line in data:
    #             s = line.strip()
    #             if not s:
    #                 contexts.append(None)
    #                 continue
    #             ctx = [PAD] * block_size
    #             for ch in s[-block_size:]:
    #                 if ch in ctoi:
    #                     ctx = ctx[1:] + [ctoi[ch]]
    #                 else:
    #                     ctx = ctx[1:] + [PAD]
    #             contexts.append(ctx)
    #         indices = [i for i, c in enumerate(contexts) if c is not None]

            
    #         for batch_start in range(0, len(indices), batch_size):
    #             batch_idx = indices[batch_start : batch_start + batch_size]
    #             x_b = torch.tensor([contexts[i] for i in batch_idx], dtype=torch.long).to(device)

    #             with torch.no_grad():
    #                 logits = model(x_b)

    #             _, top_idx = torch.topk(logits, 3, dim=-1) 

    #             for j, orig_i in enumerate(batch_idx):
    #                 chars = "".join(itoc[idx.item()] for idx in top_idx[j])
    #                 final_preds[orig_i] = chars

    #         return final_preds
    #     except Exception as e:
    #         print(f"error in run_test: {e}")
    def run_pred(self, data, work_dir):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            checkpoint = torch.load(os.path.join(work_dir, "model.pt"), map_location=device)

            ctoi         = checkpoint["ctoi"]
            itoc         = checkpoint["itoc"]
            block_size   = checkpoint["block_size"]
            vocab_size   = checkpoint["vocab_size"]
            ngram_counts = checkpoint.get("ngram_counts", None)
            ngram_order  = checkpoint.get("ngram_order", 0)
            ngram_alpha  = checkpoint.get("ngram_alpha", 1.0)

            model = CharacterLSTM(
                vocab_size=vocab_size,
                embed_dim=checkpoint["embed_dim"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()

            PAD = ctoi["."]
            batch_size = 128
            final_preds = [""] * len(data)

            contexts = []
            clean_strings = []

            for line in data:
                s = line.strip()
                clean_strings.append(s)

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
                    lstm_probs = F.softmax(logits, dim=-1).cpu()

                for j, orig_i in enumerate(batch_idx):
                    probs = lstm_probs[j]

                    if ngram_counts is not None and ngram_order > 0:
                        ngram_probs = self.get_ngram_probs(
                            clean_strings[orig_i],
                            ngram_counts,
                            ctoi,
                            vocab_size,
                            ngram_order
                        )

                        if torch.sum(ngram_probs) > 0:
                            probs = ngram_alpha * probs + (1.0 - ngram_alpha) * ngram_probs

                    _, top_idx = torch.topk(probs, 3, dim=-1)
                    chars = "".join(itoc[idx.item()] for idx in top_idx)
                    final_preds[orig_i] = chars

            return final_preds

        except Exception as e:
            print(f"error in run_test: {e}")
    
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
            print(f"error in save: {e}")

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
            print(f"error in load: {e}")
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
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('Loading training data')
        train_data = MyModel.load_training_data(train_data_dir)
        random.shuffle(train_data)

        split = int(0.9 * len(train_data))
        train_split = train_data[:split]
        val_split = train_data[split:]

        shared_cache_dir = os.path.join(args.work_dir, "shared_cache")
        os.makedirs(shared_cache_dir, exist_ok=True)

        configs = [
            {"block_size": 32, "embed_dim": 128, "hidden_size": 512, "num_layers": 2, "dropout": 0.2, "batch_size": 256, "steps": 8000,  "min_count": 1},
            {"block_size": 64, "embed_dim": 128, "hidden_size": 512, "num_layers": 2, "dropout": 0.2, "batch_size": 256, "steps": 8000,  "min_count": 1},
            {"block_size": 64, "embed_dim": 128, "hidden_size": 512, "num_layers": 2, "dropout": 0.1, "batch_size": 256, "steps": 10000, "min_count": 1},
            {"block_size": 64, "embed_dim": 128, "hidden_size": 768, "num_layers": 2, "dropout": 0.2, "batch_size": 128, "steps": 10000, "min_count": 1},
        ]

        results = []

        for i, cfg in enumerate(configs):
            print(f"\n===== Trial {i} =====")
            print(cfg)

            trial_dir = os.path.join(args.work_dir, f"trial_{i}")
            model = MyModel()

            best_val_loss = model.run_train(
                train_split,
                trial_dir,
                block_size_p=cfg["block_size"],
                embed_dim_p=cfg["embed_dim"],
                hidden_size_p=cfg["hidden_size"],
                num_layers_p=cfg["num_layers"],
                dropout_p=cfg["dropout"],
                batch_size_p=cfg["batch_size"],
                steps_p=cfg["steps"],
                min_count_p=cfg["min_count"],
                val_data=val_split,
                cache_dir=shared_cache_dir
            )

            model.save(trial_dir)

            results.append({
                "trial": i,
                "trial_dir": trial_dir,
                "best_val_loss": best_val_loss if best_val_loss is not None else float("inf"),
                "config": cfg,
            })

        print("\n" + "=" * 80)
        print("TRIAL SUMMARY")
        print("=" * 80)

        for r in results:
            print(
                f"trial {r['trial']} | "
                f"best_val_loss={r['best_val_loss']:.4f} | "
                f"dir={r['trial_dir']} | cfg={r['config']}"
            )

        best_result = min(results, key=lambda r: r["best_val_loss"])
        print("\nBEST TRIAL:")
        print(f"trial {best_result['trial']}")
        print(f"dir: {best_result['trial_dir']}")
        print(f"best val loss: {best_result['best_val_loss']:.4f}")
        print(f"config: {best_result['config']}")
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
