import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


words = open('./training_data/train_input.txt', 'r').read().split()
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
# plt.figure(figsize=(40,40))
# plt.imshow(counts_tensor, cmap='Blues')

# for i in range(85):
#     for j in range(85):
#         chstr = itoc[i] + itoc[j]
#         plt.text(j, i, str(chstr), ha='center', va='bottom', color='red')
#         plt.text(j, i, counts_tensor[i, j].item(), ha='center', va='top', color='blue')
# plt.axis('off')

# plt.show()

normalized_counts_tensor = smoothed_counts_tensor/ smoothed_counts_tensor.sum(1, keepdim=True)

log_likelihood = 0.0
samples = 0

for word in words[:10]:
    for ch1, ch2 in zip(word, word[1:]):
        idx1 = ctoi[ch1]
        idx2 = ctoi[ch2]
        prob = normalized_counts_tensor[idx1, idx2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob.item()
        samples += 1
        if samples <= 20:
            print(f"{ch1}{ch2}: {prob:.5f} | {log_prob:.5f} | {log_likelihood :.5f}")
            
print(f"{log_likelihood = }")
Negative_LL = - log_likelihood
Avg_NLL = Negative_LL/samples

print(f'Negative Log Likelihood: {Negative_LL:.5f}')
print(f'Average Negative Log Likelihood: {Avg_NLL:.5f}')

log_likelihood = 0.0
samples = 0

for word in ['Toothless']:
    for ch1, ch2 in zip(word, word[1:]):
        idx1 = ctoi[ch1]
        idx2 = ctoi[ch2]
        prob = normalized_counts_tensor[idx1, idx2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob.item()
        samples += 1
        if samples <= 20:
            print(f"{ch1}{ch2}: {prob:.5f} | {log_prob:.5f} | {log_likelihood :.5f}")
            
print(f"{log_likelihood = }")
Negative_LL = - log_likelihood
Avg_NLL = Negative_LL/samples

print(f'Negative Log Likelihood: {Negative_LL:.5f}')
print(f'Average Negative Log Likelihood: {Avg_NLL:.5f}')

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

    print(W)
    
import torch
import torch.nn.functional as F

def top3_next_chars_from_last_char(last_ch, W, ctoi, itoc):
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


# Read the example file (each line is a partial phrase)
lines = open('./example/input.txt', 'r').read().splitlines()

for line in lines:
    s = line.rstrip("\n")
    if len(s) == 0:
        print("(empty line) -> can't predict")
        continue

    last_ch = s[-1]
    preds = top3_next_chars_from_last_char(last_ch, W, ctoi, itoc)

    # Pretty print
    pred_str = ", ".join([f"'{ch}' ({p:.3f})" for ch, p in preds])
    print(f"{s!r} | last='{last_ch}' -> top3: {pred_str}")


