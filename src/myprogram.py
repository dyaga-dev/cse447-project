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
        # your code here
        # this particular model doesn't train
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        words = [word for s in data for word in s.split()]
        # word len : 171914
        # min and max later

        ctoi = {ch : i for i, ch in enumerate(sorted(set("".join(words))))}
        itoc = {v:k for k, v in ctoi.items()}

        chars = sorted(list(set("".join(words))))
        chars = ['.'] + chars          # start / stop token
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
        num = X.shape[0]

        embed_dim = 10
        hidden1 = 100
        hidden2 = 100
        gen = torch.Generator().manual_seed(2147483647)
        embd = torch.randn((vocab_size, embed_dim), generator=gen, requires_grad=True)

        W1 = torch.randn((block_size * embed_dim, hidden1), generator=gen, requires_grad=True)
        B1 = torch.zeros(hidden1, requires_grad=True)

        W2 = torch.randn((hidden1, hidden2), generator=gen, requires_grad=True)
        B2 = torch.zeros(hidden2, requires_grad=True)

        W3 = torch.randn((hidden2, vocab_size), generator=gen, requires_grad=True)
        B3 = torch.zeros(vocab_size, requires_grad=True)

        parameters = [embd, W1, B1, W2, B2, W3, B3]

        lr = 0.01
        steps = 5000
        batch_size = 64

        for step in range(steps):
            idx = torch.randint(0, X.shape[0], (batch_size,))

            emb = embd[X[idx]]                 # (B, 3, 10)
            h = emb.view(batch_size, -1)       # (B, 30)

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

        torch.save({
            "embd": embd,
            "W1": W1, "B1": B1,
            "W2": W2, "B2": B2,
            "W3": W3, "B3": B3,
            "ctoi": ctoi,
            "itoc": itoc,
            "block_size": block_size
        }, os.path.join(work_dir, "model.pt"))
        

    def run_pred(self, data, work_dir):
        # your code here
        checkpoint = torch.load(os.path.join(work_dir, "model.pt"))

        embd = checkpoint["embd"]
        W1, B1 = checkpoint["W1"], checkpoint["B1"]
        W2, B2 = checkpoint["W2"], checkpoint["B2"]
        W3, B3 = checkpoint["W3"], checkpoint["B3"]
        ctoi = checkpoint["ctoi"]
        itoc = checkpoint["itoc"]
        block_size = checkpoint["block_size"]

        final_preds = []

        for line in data:
            s = line.rstrip("\n")
            if len(s) == 0:
                print("(empty line) -> can't predict")
                final_preds.append("")
                continue

            context = [ctoi["."]] * block_size
            for ch in s[-block_size:]:
                if ch in ctoi:
                    context = context[1:] + [ctoi[ch]]

            x = torch.tensor([context])
            emb = embd[x]                 # (1, block_size, embed_dim)
            h = emb.view(1, -1)
            h1 = torch.tanh(h @ W1 + B1)
            h2 = torch.tanh(h1 @ W2 + B2)
            logits = h2 @ W3 + B3
            probs = F.softmax(logits, dim=1).squeeze(0)

            top_probs, top_idx = torch.topk(probs, 3)
            preds = [(itoc[i.item()], top_probs[j].item()) for j, i in enumerate(top_idx)]

            final_preds.append("".join([ch for ch, _ in preds]))

            pred_str = ", ".join([f"'{ch}' ({p:.3f})" for ch, p in preds])
            print(f"{s!r} -> top3: {pred_str}")
        return final_preds

    def save(self, work_dir):
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

    @classmethod
    def load(cls, work_dir):
        model = cls()
        path = os.path.join(work_dir, "model.checkpoint")
        with open(path, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        # Only restore args if you actually want them
        # (and if your model has/needs an args object).
        return model



if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    train_data_file = 'training_data/train_input.txt'

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
