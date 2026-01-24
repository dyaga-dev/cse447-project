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

    def top3_next_chars_from_last_char(self, last_ch, W, ctoi, itoc):
        # If last character isn't in vocab, you can handle it here
        if last_ch not in ctoi:
            return [("<?>", 0.0), ("<?>", 0.0), ("<?>", 0.0)]

        idx = ctoi[last_ch]

        # (1, V)
        enc_x = F.one_hot(torch.tensor([idx]), num_classes=W.shape[0]).float()
        logits = enc_x @ W  # (1, V)

        # Convert logits -> probabilities
        probs = logits.softmax(dim=1).squeeze(0)  # (V,)

        # Top 3 indices
        top_probs, top_idx = torch.topk(probs, k=3)

        # Return [(char, prob), ...]
        return [(itoc[i.item()], top_probs[j].item()) for j, i in enumerate(top_idx)]
    
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
        words = words = [word for s in data for word in s.split()]
        # word len : 171914
        # min and max later

        ctoi = {ch : i for i, ch in enumerate(sorted(set("".join(words))))}
        itoc = {v:k for k, v in ctoi.items()}

        counts_tensor = torch.zeros((85,85), dtype=torch.int32)

        for word in words:
            for ch1, ch2 in zip(word, word[1:]):
                idx1 = ctoi[ch1]
                idx2 = ctoi[ch2]
                counts_tensor[idx1, idx2] += 1

        # bigram model
        smoothed_counts_tensor = (counts_tensor + 1)

        normalized_counts_tensor = smoothed_counts_tensor/ smoothed_counts_tensor.sum(1, keepdim=True)

        x, y = [], []

        for word in words:
            for ch1, ch2 in zip(word, word[1:]):
                idx1 = ctoi[ch1]
                idx2 = ctoi[ch2]
                x.append(idx1)
                y.append(idx2)
        X = torch.tensor(x)
        Y = torch.tensor(y)
        num = X.nelement()

        gen = torch.Generator().manual_seed(2147483647)

        W = torch.randn((85, 85), generator=gen, requires_grad=True)
        enc_x = F.one_hot(X, num_classes=85).float()
        logits = enc_x @ W
        counts = logits.exp()
        prob = counts/counts.sum(1, keepdim=True)
        loss = -prob[torch.arange(num), Y].log().mean() + 0.01*(W**2).mean()

        lre = torch.linspace(-3, 0, 1000)
        lrs = 10 ** lre

        lri = []
        lossi = []
        print("line 106")
        for i in range(10):
            enc_x = F.one_hot(X, num_classes=85).float()
            logits = enc_x @ W
            counts = logits.exp()
            prob = counts/counts.sum(1, keepdim=True)
            loss = -prob[torch.arange(num), Y].log().mean() + 0.01*(W**2).mean()
                
            loss.backward()

            lr = lrs[i]
            
            # update parameters
            W.data += -lr * W.grad
                
            # track
            lri.append(lre[i])
            lossi.append(loss.item())

        lossi_np = np.array(lossi)
        min_idx = lossi_np.argmin()
        min_loss = lossi_np[min_idx]
        best_lre = lri[min_idx]
        best_lr = 10 ** best_lre
        print(f'best_ lr: {best_lr:.5f}')

        steps = 100
        for step in range(steps):
            enc_x = F.one_hot(X, num_classes=85).float()
            logits = enc_x @ W
            counts = logits.exp()
            prob = counts/counts.sum(1, keepdim=True)
            loss = -prob[torch.arange(num), Y].log().mean() + 0.01*(W**2).mean()
            
            print(f'Step: {step}, Loss: {loss.item()}')
            
            W.grad = None
            loss.backward()
            W.data += -best_lr * W.grad
            
            ctoiFile = os.path.join(work_dir, "ctoi.txt")
            with open(ctoiFile, "w") as f:
                for c, i in ctoi.items():
                    f.write(f"{repr(c)}\t{i}\n")
            
            itocFile = os.path.join(work_dir, "itoc.txt")
            with open(itocFile, "w") as f:
                for i, c in itoc.items():
                    f.write(f"{i}\t{repr(c)}\n")

            wFile = os.path.join(work_dir, "W.pt")
            torch.save(W, wFile)
        

    def run_pred(self, data):
        # your code here
        ctoiFile = os.path.join("work", "ctoi.txt")
        ctoi = {}

        with open(ctoiFile, "r") as f:
            for line in f:
                c_repr, i = line.rstrip("\n").split("\t")
                c = eval(c_repr)        # converts "'a'" → "a"
                ctoi[c] = int(i)
        
        itocFile = os.path.join("work", "itoc.txt")
        itoc = {}

        with open(itocFile, "r") as f:
            for line in f:
                i_str, c_repr = line.rstrip("\n").split("\t")
                i = int(i_str)
                c = eval(c_repr)   # "'a'" → "a", "'\n'" → "\n"
                itoc[i] = c
        
        W = torch.load("./work/W.pt")

        final_preds = []
        lines = data

        for line in lines:
            s = line.rstrip("\n")
            if len(s) == 0:
                print("(empty line) -> can't predict")
                continue

            last_ch = s[-1]
            preds = self.top3_next_chars_from_last_char(last_ch, W, ctoi, itoc)

            # Pretty print
            pred_str = ", ".join([f"'{ch}' ({p:.3f})" for ch, p in preds])
            print(f"{s!r} | last='{last_ch}' -> top3: {pred_str}")

            pred_str = "".join([f"{ch}" for ch, _ in preds])
            final_preds.append(pred_str)
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
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
