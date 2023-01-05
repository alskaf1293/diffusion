import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import math

IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    plt.show()

data = load_transformed_dataset()
#print(data.shape)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getLinearBetaSchedule(start=0.0001, end=0.02, steps=300):
    return torch.linspace(start, end, steps)

def getIndex(list, indicies, imgShape):

    batchSize = indicies.shape[0]
    values = list.gather(-1, indicies.cpu())

    return values.reshape(batchSize, *((1,)*(len(imgShape)-1))).to(device)

def forwardDiffusion(img, t, localDevice="cpu"):
    #Applys forward diffusion given an image img and timestep t
    #img shape: (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    #t shape: (BATCH_SIZE)

    sqrtCumprodAlphas_t = getIndex(sqrtCumprodAlphas,t,img.shape)
    sqrtOneMinusCumprodAlphas_t = getIndex(sqrtOneMinusCumprodAlphas,t, img.shape)

    noise = torch.randn_like(img)

    return sqrtCumprodAlphas_t.to(localDevice) * img.to(localDevice) + noise.to(localDevice) * sqrtOneMinusCumprodAlphas_t.to(localDevice), noise.to(localDevice)

T = 300
betas = getLinearBetaSchedule(steps=T)

alphas = 1. - betas
cumprodAlphas = torch.cumprod(alphas, axis=0)
cumprodAlphasOne = F.pad(cumprodAlphas[:-1], (1,0) , value=1.0)
sqrtRecipAlphas = torch.sqrt(1.0 / alphas)
sqrtCumprodAlphas = torch.sqrt(cumprodAlphas)
sqrtOneMinusCumprodAlphas = torch.sqrt(1. - cumprodAlphas)

posteriorVariance = betas * (1. - cumprodAlphasOne) / (1. - cumprodAlphas)

"""
T = 300
# Simulate forward diffusion
# next(iter(dataloader)) shape: [(BATCH_SIZE, 3, 64, 64), (BATCH_SIZE)]
# image shape: BATCH_SIZE, 3, 64, 64
image = next(iter(dataloader))[0]
#print(next(iter(dataloader))[1].shape)
plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, int(num_images+1), int((idx/stepsize) + 1))
    image, noise = forwardDiffusion(image, t)
    show_tensor_image(image)
"""

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

model = SimpleUnet().to(device)

def getLoss(model, img, t):
    forward, noise = forwardDiffusion(img, t)
    forward = forward.to(device)
    noise = noise.to(device)
    prediction = model(forward, t)
    return F.l1_loss(prediction,noise)

@torch.no_grad()
def sample(img, t):
    if t > 1:
        z = torch.randn_like(img)
    else:
        z = torch.zeros_like(img)
    
    alphas_t = getIndex(alphas, t, img.shape)
    sqrtOneMinusCumprodAlphas_t = getIndex(sqrtOneMinusCumprodAlphas, t, img.shape)
    sqrtRecipAlphas_t = getIndex(sqrtRecipAlphas, t, img.shape)
    posteriorVariance_t = getIndex(posteriorVariance, t, img.shape)

    return sqrtRecipAlphas_t * (img - ( (1 - alphas_t)/ sqrtOneMinusCumprodAlphas_t)*model(img, t)) + posteriorVariance_t * z

@torch.no_grad()
def plotFromNoise():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    #plt.figure(figsize=(15,15))
    #plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample(img, t)
        #if i % stepsize == 0:
        #    plt.subplot(1, int(num_images), int(i/stepsize+1))
    show_tensor_image(img.detach().cpu())

from torch.optim import Adam

model = model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 500

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = getLoss(model, batch[0].to(device), t.to(device))

        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            plotFromNoise()