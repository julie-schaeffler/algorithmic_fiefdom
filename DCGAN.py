import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

class SealDataset(Dataset):
    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(json_file, 'r', encoding='utf-8') as f:
            self.code_dict = json.load(f)
        self.keys = sorted(list(self.code_dict.keys()), key=lambda x: int(x))
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = os.path.join(self.image_dir, f"{key}.png")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        code_text = self.code_dict[key]
        return image, code_text

def build_vocab(codes, min_freq=1):
    from collections import Counter
    counter = Counter()
    for code in codes:
        tokens = code.lower().split()
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def tokenize(text, vocab, max_len=20):
    tokens = text.lower().split()
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(token_ids) < max_len:
        token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

image_dir = "processed_cv"     
json_file = "gan_text_components_shuffled.json"   

base_dataset = SealDataset(image_dir, json_file, transform)
codes_list = [code for _, code in base_dataset]
vocab = build_vocab(codes_list)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

class TokenizedSealDataset(Dataset):
    def __init__(self, base_dataset, vocab, max_seq_len):
        self.base_dataset = base_dataset
        self.vocab = vocab
        self.max_seq_len = max_seq_len
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, code_text = self.base_dataset[idx]
        token_ids = tokenize(code_text, self.vocab, self.max_seq_len)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        return image, token_ids

max_seq_len = 20
dataset = TokenizedSealDataset(base_dataset, vocab, max_seq_len)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embeds = self.embedding(x)  
        _, (h_n, _) = self.lstm(embeds) 
        return h_n.squeeze(0) 

class ConditionalGenerator(nn.Module):
    def __init__(self, nz, text_hidden_size, ngf, nc):
        super(ConditionalGenerator, self).__init__()
        self.fc = nn.Linear(nz + text_hidden_size, ngf * 8 * 4 * 4)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  
        )
    
    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        img = self.main(x)
        return img

class ConditionalDiscriminator(nn.Module):
    def __init__(self, nc, ndf, text_hidden_size):
        super(ConditionalDiscriminator, self).__init__()
        self.image_net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Linear(ndf * 8 * 4 * 4 + text_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image, text_embedding):
        x = self.image_net(image)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, text_embedding), dim=1)
        validity = self.sigmoid(self.fc(x))
        return validity


nz = 100               
ngf = 64               
ndf = 64               
nc = 3                 
text_embed_size = 128  
text_hidden_size = 128 
lr = 0.0002
beta1 = 0.5
num_epochs = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = TextEncoder(vocab_size, text_embed_size, text_hidden_size).to(device)
netG = ConditionalGenerator(nz, text_hidden_size, ngf, nc).to(device)
netD = ConditionalDiscriminator(nc, ndf, text_hidden_size).to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)
text_encoder.apply(weights_init)

criterion = nn.BCELoss()

optimizerD = optim.Adam(list(netD.parameters()) + list(text_encoder.parameters()), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


real_label = 1.
fake_label = 0.

for epoch in range(num_epochs):
    for i, (images, token_ids) in enumerate(dataloader):
        current_batch = images.size(0)
        images = images.to(device)
        token_ids = token_ids.to(device)
        
        text_features = text_encoder(token_ids)
        
        netD.zero_grad()

        label_real = torch.full((current_batch,), real_label, dtype=torch.float, device=device)
        label_fake = torch.full((current_batch,), fake_label, dtype=torch.float, device=device)

        output_real = netD(images, text_features)
        errD_real = criterion(output_real.view(-1), label_real.view(-1))

        noise = torch.randn(current_batch, nz, device=device)
        fake_images = netG(noise, text_features)
        output_fake = netD(fake_images.detach(), text_features)
        errD_fake = criterion(output_fake.view(-1), label_fake.view(-1))

        errD_total = errD_real + errD_fake
        errD_total.backward()
        optimizerD.step()

        netG.zero_grad()
        fake_images = netG(noise, text_features.detach())
        output = netD(fake_images, text_features.detach())
        errG = criterion(output.view(-1), label_real.view(-1))  
        errG.backward()
        optimizerG.step()

        
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] " \
                  f"Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f}")
    
    with torch.no_grad():
        fixed_noise = torch.randn(64, nz, device=device)
        fixed_text_features = text_features[0].unsqueeze(0).expand(64, -1).detach()
        fake = netG(fixed_noise, fixed_text_features)
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        npimg = grid.cpu().numpy()
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.savefig(f"epochs/output_epoch_{epoch}.png")  
        plt.close() 

torch.save(netG.state_dict(), "netG.pth")
torch.save(text_encoder.state_dict(), "text_encoder.pth")
