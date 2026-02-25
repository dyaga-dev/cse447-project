#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
import torch.nn.functional as F
import os


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    
    @classmethod
    def load_training_data(cls, fname):
        """
        Backwards compatible:
        - If fname is a file: read lines as before.
        - If fname is a directory: read and concatenate all train_input_*.txt files inside.
        """
        try:
            data = []
            if os.path.isdir(fname):
                # read all per-language training inputs
                files = sorted([
                    os.path.join(fname, fn)
                    for fn in os.listdir(fname)
                    if fn.startswith("train_input_") and fn.endswith(".txt")
                ])
                if not files:
                    raise ValueError(f"No train_input_*.txt files found in directory: {fname}")

                for fp in files:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            data.append(line.rstrip("\n"))
                return data

            # else: assume it's a single file path
            with open(fname, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    data.append(line.rstrip("\n"))
            return data

        except Exception as e:
            print("error in load_training_data: " + str(e))
            return []

    @classmethod
    def load_test_data(cls, path):
        """
        Backwards compatible:
        - If `path` is a file: read it (original grader behavior).
        - If `path` is a directory: read and concatenate all test_input_*.txt inside.
        (good for local multilingual testing)
        """
        data = []

        # Directory case (your local testing)
        if os.path.isdir(path):
            files = sorted([
                os.path.join(path, fn)
                for fn in os.listdir(path)
                if fn.startswith("test_input_") and fn.endswith(".txt")
            ])
            if not files:
                raise ValueError(f"No test_input_*.txt files found in directory: {path}")

            for fp in files:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        data.append(line.rstrip("\n"))
            return data

        # File case (prof hidden test)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                data.append(line.rstrip("\n"))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wt') as f:
                for p in preds:
                    f.write(f"{p}\n")
        except Exception as e:
            print("error in write_pred:", e)

    # def run_train(self, data, work_dir):
    #     try:
    #         words = [word for s in data for word in s.split()]
    #         # word len : 171914
    #         # min and max later

    #         ctoi = {ch : i for i, ch in enumerate(sorted(set("".join(words))))}
    #         itoc = {v:k for k, v in ctoi.items()}

    #         chars = sorted(list(set("".join(words))))
    #         if '.' in chars:
    #             chars.remove('.')
    #         chars = ['.'] + chars          # start / stop token
    #         ctoi = {ch:i for i,ch in enumerate(chars)}
    #         itoc = {i:ch for ch,i in ctoi.items()}
    #         vocab_size = len(chars)

    #         block_size = 3

    #         x, y = [], []

    #         for word in words:
    #             context = [ctoi["."]] * block_size
    #             for ch in word:
    #                 idx = ctoi[ch]
    #                 x.append(context)
    #                 y.append(idx)
    #                 context = context[1:] + [idx]
    #         X = torch.tensor(x)
    #         Y = torch.tensor(y)
    #         num = X.shape[0]

    #         embed_dim = 10
    #         hidden1 = 100
    #         hidden2 = 100
    #         gen = torch.Generator().manual_seed(2147483647)
    #         embd = torch.randn((vocab_size, embed_dim), generator=gen, requires_grad=True)

    #         W1 = torch.randn((block_size * embed_dim, hidden1), generator=gen, requires_grad=True)
    #         B1 = torch.zeros(hidden1, requires_grad=True)

    #         W2 = torch.randn((hidden1, hidden2), generator=gen, requires_grad=True)
    #         B2 = torch.zeros(hidden2, requires_grad=True)

    #         W3 = torch.randn((hidden2, vocab_size), generator=gen, requires_grad=True)
    #         B3 = torch.zeros(vocab_size, requires_grad=True)

    #         parameters = [embd, W1, B1, W2, B2, W3, B3]

    #         lr = 0.01
    #         steps = 5000
    #         batch_size = 64

    #         for step in range(steps):
    #             idx = torch.randint(0, X.shape[0], (batch_size,))

    #             emb = embd[X[idx]]                 # (B, 3, 10)
    #             h = emb.view(batch_size, -1)       # (B, 30)

    #             h1 = torch.tanh(h @ W1 + B1)
    #             h2 = torch.tanh(h1 @ W2 + B2)
    #             logits = h2 @ W3 + B3

    #             loss = F.cross_entropy(logits, Y[idx])

    #             for p in parameters:
    #                 p.grad = None
    #             loss.backward()

    #             for p in parameters:
    #                 p.data += -lr * p.grad

    #             if step % 250 == 0:
    #                 print(f"step {step} | loss {loss.item():.4f}")
    #         # print(itoc)
    #         torch.save({
    #             "embd": embd,
    #             "W1": W1, "B1": B1,
    #             "W2": W2, "B2": B2,
    #             "W3": W3, "B3": B3,
    #             "ctoi": ctoi,
    #             "itoc": itoc,
    #             "block_size": block_size
    #         }, os.path.join(work_dir, "model.pt"))
    #     except Exception as e:
    #         print("error in run_test: " + e)
        

    # def run_pred(self, data, work_dir):
    #     # your code here
    #     try:
    #         checkpoint = torch.load(os.path.join(work_dir, "model.pt"), map_location="cpu")

    #         embd = checkpoint["embd"]
    #         W1, B1 = checkpoint["W1"], checkpoint["B1"]
    #         W2, B2 = checkpoint["W2"], checkpoint["B2"]
    #         W3, B3 = checkpoint["W3"], checkpoint["B3"]
    #         ctoi = checkpoint["ctoi"]
    #         itoc = checkpoint["itoc"]
    #         block_size = checkpoint["block_size"]

    #         final_preds = []
    #         for line in data:
    #             s = line.rstrip("\n")
    #             if len(s) == 0:
    #                 print("(empty line) -> can't predict")
    #                 final_preds.append("")
    #                 continue
                
    #             context = [ctoi["."]] * block_size
    #             for ch in s[-block_size:]:
    #                 if ch in ctoi:
    #                     context = context[1:] + [ctoi[ch]]

    #             x = torch.tensor([context])
    #             emb = embd[x]                 # (1, block_size, embed_dim)
    #             h = emb.view(1, -1)
    #             h1 = torch.tanh(h @ W1 + B1)
    #             h2 = torch.tanh(h1 @ W2 + B2)
    #             logits = h2 @ W3 + B3
    #             probs = F.softmax(logits, dim=1).squeeze(0)

    #             top_probs, top_idx = torch.topk(probs, 3)
    #             preds = [(itoc[i.item()], top_probs[j].item()) for j, i in enumerate(top_idx)]

    #             final_preds.append("".join([ch for ch, _ in preds]))

    #             pred_str = ", ".join([f"'{ch}' ({p:.3f})" for ch, p in preds])
    #             print(f"{s!r} -> top3: {pred_str}")
    #         return final_preds
    #     except Exception as e:
    #         print("error in run_test: " + e)

    def run_train(self, data, work_dir):
        try:
            words = [word for s in data for word in s.split()]

            # --- neural net setup unchanged ---
            chars = sorted(list(set("".join(words))))
            if '.' in chars:
                chars.remove('.')
            chars = ['.'] + chars
            ctoi = {ch:i for i,ch in enumerate(chars)}
            itoc = {i:ch for ch,i in ctoi.items()}
            vocab_size = len(chars)
            block_size = 3

            x, y = [], []
            for word in words:
                context = [ctoi["."]] * block_size
                for ch in word:
                    idx = ctoi[ch]
                    x.append(context)
                    y.append(idx)
                    context = context[1:] + [idx]
            X = torch.tensor(x)
            Y = torch.tensor(y)

            embed_dim = 10
            hidden1, hidden2 = 100, 100
            gen = torch.Generator().manual_seed(2147483647)
            embd = torch.randn((vocab_size, embed_dim), generator=gen, requires_grad=True)
            W1 = torch.randn((block_size*embed_dim, hidden1), generator=gen, requires_grad=True)
            B1 = torch.zeros(hidden1, requires_grad=True)
            W2 = torch.randn((hidden1, hidden2), generator=gen, requires_grad=True)
            B2 = torch.zeros(hidden2, requires_grad=True)
            W3 = torch.randn((hidden2, vocab_size), generator=gen, requires_grad=True)
            B3 = torch.zeros(vocab_size, requires_grad=True)
            parameters = [embd, W1, B1, W2, B2, W3, B3]

            lr, steps, batch_size = 0.01, 5000, 64

            for step in range(steps):
                idx = torch.randint(0, X.shape[0], (batch_size,))
                emb = embd[X[idx]]
                h = emb.view(batch_size, -1)
                h1 = torch.tanh(h @ W1 + B1)
                h2 = torch.tanh(h1 @ W2 + B2)
                logits = h2 @ W3 + B3
                loss = F.cross_entropy(logits, Y[idx])
                for p in parameters:
                    p.grad = None
                loss.backward()
                for p in parameters:
                    p.data += -lr * p.grad
                if step % 250 == 0:
                    print(f"step {step} | loss {loss.item():.4f}")

            # --- build n-gram counts for Kneser-Ney ---
            ngram_counts, context_counts = self._build_ngram_counts(words, block_size)
            torch.save({
                "embd": embd, "W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3,
                "ctoi": ctoi, "itoc": itoc, "block_size": block_size,
                "ngram_counts": ngram_counts, "context_counts": context_counts
            }, os.path.join(work_dir, "model.pt"))
        except Exception as e:
            print("error in run_train:", e)

    def _build_ngram_counts(self, words, n):
        """Build n-gram counts for Kneser-Ney."""
        ngram_counts = {}
        context_counts = {}
        for word in words:
            padded = ['.']*n + list(word)
            for i in range(n, len(padded)):
                ctx = tuple(padded[i-n:i])
                ch = padded[i]
                ngram_counts[(ctx, ch)] = ngram_counts.get((ctx, ch), 0) + 1
                context_counts[ctx] = context_counts.get(ctx, 0) + 1
        return ngram_counts, context_counts

    def _kneser_ney_prob(self, context, ch, ngram_counts, context_counts, discount=0.75):
        """Compute P_KN(ch | context) using Kneser-Ney smoothing."""
        if context in context_counts and (context, ch) in ngram_counts:
            count = ngram_counts[(context, ch)]
            total = context_counts[context]
            return max(count - discount, 0)/total
        elif context in context_counts:
            # back-off weight
            return discount * len([c for (ctx, c) in ngram_counts if ctx==context])/context_counts[context] * 1e-6
        else:
            # unseen context
            return 1e-6

    def run_pred(self, data, work_dir):
        try:
            checkpoint = torch.load(os.path.join(work_dir, "model.pt"), map_location="cpu")
            embd = checkpoint["embd"]
            W1, B1 = checkpoint["W1"], checkpoint["B1"]
            W2, B2 = checkpoint["W2"], checkpoint["B2"]
            W3, B3 = checkpoint["W3"], checkpoint["B3"]
            ctoi, itoc = checkpoint["ctoi"], checkpoint["itoc"]
            block_size = checkpoint["block_size"]
            ngram_counts = checkpoint["ngram_counts"]
            context_counts = checkpoint["context_counts"]

            final_preds = []
            for line in data:
                s = line.rstrip("\n")
                if len(s) == 0:
                    final_preds.append("")
                    continue
                context = [ctoi["."]] * block_size
                for ch in s[-block_size:]:
                    if ch in ctoi:
                        context = context[1:] + [ctoi[ch]]
                x = torch.tensor([context])
                emb = embd[x]
                h = emb.view(1, -1)
                h1 = torch.tanh(h @ W1 + B1)
                h2 = torch.tanh(h1 @ W2 + B2)
                logits = h2 @ W3 + B3
                probs = F.softmax(logits, dim=1).squeeze(0)

                # --- integrate Kneser-Ney fallback ---
                kn_probs = []
                for i in range(len(probs)):
                    ch_i = itoc[i]
                    kn_prob = self._kneser_ney_prob(tuple(context), ch_i, ngram_counts, context_counts)
                    kn_probs.append(kn_prob)
                kn_probs = torch.tensor(kn_probs)
                final_probs = probs + 0.1*kn_probs  # small weight for KN smoothing
                final_probs /= final_probs.sum()

                top_probs, top_idx = torch.topk(final_probs, 3)
                preds = [(itoc[i.item()], top_probs[j].item()) for j, i in enumerate(top_idx)]
                final_preds.append("".join([ch for ch, _ in preds]))

                pred_str = ", ".join([f"'{ch}' ({p:.3f})" for ch, p in preds])
                print(f"{s!r} -> top3: {pred_str}")
            return final_preds
        except Exception as e:
            print("error in run_pred:", e)

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

    # Prefer multilingual folder if present; fall back to original single-file training.
    # train_data_file = 'multilingual_dataset' if os.path.isdir('multilingual_dataset') else 
    train_data_file = 'multilingual_dataset' if os.path.isdir('multilingual_dataset') else 'multilingual_dataset/train_input_eng.txt'

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data(train_data_file)
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
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
