import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
seed = 284795
torch.manual_seed(seed)
np.random.seed(seed)

def load_photos_as_arrays():
    load_folder = "processed2/"
    photo_arrays = []
    for filename in os.listdir(load_folder):
        if filename.endswith('.png'):
            # Read image in grayscale
            img = cv2.imread(os.path.join(load_folder, filename), cv2.IMREAD_GRAYSCALE)
            photo_arrays.append(img)
    return np.array(photo_arrays)

# Hyperparameters for training loop
n_epochs = 80
noise_dimension = 64
lr = 0.0005
display_step = 150
batch_size = 16
hidden_dimension = 128
gan_shape = [8, 16, 16, 8] #zakladamy 4 warstwy
saveFolder = (f"generated_images_,gan_shape={gan_shape},hd={hidden_dimension} bs={batch_size},lr={lr},noise={noise_dimension}")
os.makedirs(saveFolder, exist_ok=True)

# Define a function to save generated images
def save_generated_images(images, epoch):
    for idx, img in enumerate(images):
        # Detach the tensor from the computation graph and convert it to a numpy array
        image_unflat = img.detach().cpu().view(-1, * (1, 128, 128))
        img_name = f"generated_epoch{epoch}_img{idx}.png"
        os.makedirs(f"{saveFolder}/generated_epoch{epoch}", exist_ok=True)
        img_path = os.path.join(f"{saveFolder}/generated_epoch{epoch}", img_name)

        # Save the tensor as an image
        save_image(image_unflat, img_path)



photos = load_photos_as_arrays()
print("Number of photos loaded:", len(photos))
photos_tensor = torch.tensor(photos, dtype=torch.float32)
dataloader = DataLoader(
    photos_tensor,
    batch_size=batch_size,
    shuffle=True
)

class Generator(nn.Module):
    def __init__(self, noise_dimension,
                 image_dimension=128*128,
                 hidden_dimension=hidden_dimension):
        super(Generator, self).__init__()

        self.n_dim = noise_dimension
        self.im_dim = image_dimension
        self.h_dim = hidden_dimension

        # Generator network
        self.gen = nn.Sequential(
            self.generator_block(self.n_dim, self.h_dim * gan_shape[0]),
            self.generator_block(self.h_dim * gan_shape[0], self.h_dim * gan_shape[1]),
            self.generator_block(self.h_dim * gan_shape[1], self.h_dim * gan_shape[2]),
            self.generator_block(self.h_dim * gan_shape[2], self.h_dim * gan_shape[3]),
            nn.Linear(self.h_dim * gan_shape[3], self.im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    # Simple neural network single block
    def generator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.BatchNorm1d(out_dimension),
            nn.ReLU(inplace=True),
        )
def get_noise(n_samples, noise_vector_dimension, device='cpu'):
    return torch.randn(n_samples, noise_vector_dimension,device=device)

class Discriminator(nn.Module):
    def __init__(self, image_dimension= 128 * 128,
                 hidden_dimension=hidden_dimension):
        super(Discriminator, self).__init__()

        self.im_dim = image_dimension
        self.h_dim = hidden_dimension
    #   Discriminator network
        self.disc = nn.Sequential(
            self.discriminator_block(self.im_dim, self.h_dim * gan_shape[0]),
            self.discriminator_block(self.h_dim * gan_shape[0], self.h_dim * gan_shape[1]),
            self.discriminator_block(self.h_dim * gan_shape[1], self.h_dim * gan_shape[2]),
            self.discriminator_block(self.h_dim * gan_shape[2], self.h_dim * gan_shape[3]),
            nn.Linear(self.h_dim * gan_shape[3], 1)
        )

    def forward(self, image):
        return self.disc(image)

    def discriminator_block(self, in_dimension, out_dimension):
        return nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.LeakyReLU(0.2, inplace=True)
        )

criterion = nn.BCEWithLogitsLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generator & Optimizer for Generator
gen = Generator(noise_dimension).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

# Discriminator & Optimizer for Discriminator
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, noise_dimension, device):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # All of them will got label as 0
    # .detach() here is to ensure that only discriminator parameters will get update
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred,
                               torch.zeros_like(disc_fake_pred))

    # Pass real features to discriminator
    # All of them will got label as 1
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred,
                               torch.ones_like(disc_real_pred))

    # Average of loss from both real and fake features
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, noise_dimension, device):
    # Generate noise and pass to generator
    fake_noise = get_noise(num_images, noise_dimension, device=device)
    fake = gen(fake_noise)

    # Pass fake features to discriminator
    # But all of them will got label as 1
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss = False

for epoch in range(n_epochs):
    for real in tqdm(dataloader):
        # Get number of batch size (number of image)
        # And get tensor for each image in batch
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        ### Traing discriminator ###
        # Zero out the gradient .zero_grad()
        # Calculate discriminator loss get_disc_loss()
        # Update gradient .gradient()
        # Update optimizer .step()
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, noise_dimension, device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        ### Traing generator ###
        # Zero out the gradient .zero_grad()
        # Calculate discriminator loss get_gen_loss()
        # Update gradient .gradient()
        # Update optimizer .step()
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, noise_dimension, device)
        gen_loss.backward()
        gen_opt.step()

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        if cur_step % display_step == 0 and cur_step > 0:

            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")

            fake_noise = get_noise(cur_batch_size, noise_dimension, device=device)
            fake = gen(fake_noise)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
            save_generated_images(fake, epoch)
            # Save generator model
            torch.save(gen.state_dict(), f'{saveFolder}/generated_epoch{epoch}/generator_epoch_{epoch}_step_{cur_step}.pth')

            # Save discriminator model
            torch.save(disc.state_dict(), f'{saveFolder}/generated_epoch{epoch}/discriminator_epoch_{epoch}_step_{cur_step}.pth')
        cur_step += 1