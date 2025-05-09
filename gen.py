import torch
import json
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 4742       
text_embed_size = 128  
text_hidden_size = 128  
nz = 100                
ngf = 64               
nc = 3                  

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(TextEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        return h_n.squeeze(0)

class ConditionalGenerator(torch.nn.Module):
    def __init__(self, nz, text_hidden_size, ngf, nc):
        super(ConditionalGenerator, self).__init__()
        self.fc = torch.nn.Linear(nz + text_hidden_size, ngf * 8 * 4 * 4)
        self.main = torch.nn.Sequential(
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh() 
        )
    
    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        img = self.main(x)
        return img

text_encoder = TextEncoder(vocab_size, text_embed_size, text_hidden_size).to(device)
netG = ConditionalGenerator(nz, text_hidden_size, ngf, nc).to(device)

text_encoder.load_state_dict(torch.load("text_encoder.pth", map_location=device))
netG.load_state_dict(torch.load("netG.pth", map_location=device))
text_encoder.eval()
netG.eval()

code_text = "Facebook: The service can use follow -up technologies in third -party websites to follow your activity online while you do not visit their site."

def tokenize(text, vocab, max_len=20):
    tokens = text.lower().split()
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(token_ids) < max_len:
        token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
max_len = 20
token_ids = tokenize(code_text, vocab, max_len)
token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    text_feature = text_encoder(token_ids)

noise = torch.randn(1, nz, device=device)

with torch.no_grad():
    fake_image = netG(noise, text_feature)

fake_image = (fake_image + 1) / 2.0

fake_image_np = fake_image.squeeze(0).cpu().permute(1, 2, 0).numpy()

fake_image_np = (fake_image_np * 255).astype(np.uint8)

target_width = 1000
target_height = 1000

upscaled_image = cv2.resize(fake_image_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

gray_upscaled = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2GRAY)

edges_upscaled = cv2.Canny(gray_upscaled, threshold1=35, threshold2=10)

kernel = np.ones((5, 5), np.uint8)
edges_thick = cv2.dilate(edges_upscaled, kernel, iterations=1)

cv2.imwrite("generated_edge_image_upscaled.png", edges_thick)
cv2.imwrite("generated_image_upscaled.png", upscaled_image)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("upscaled image")
plt.imshow(upscaled_image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("edge filter on upscaled image")
plt.imshow(edges_thick, cmap='gray')
plt.axis("off")
plt.show()