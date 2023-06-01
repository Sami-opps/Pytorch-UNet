import torch
from PIL import Image

# Load the image
img = Image.open("C:\\Users\\samue\\Downloads\\Senior_Project\\Pytorch-UNet\\Black_white.jpg")

# Convert the image to a PyTorch tensor
tensor_img = torch.Tensor(
    torch.ByteTensor(
        torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0])
)

# Create a black tensor of the same size as the original image
black_tensor = torch.zeros_like(tensor_img)

# Set all non-black pixels to black
black_tensor[tensor_img != 0] = 0

# Save the black image
black_img = Image.fromarray(black_tensor.numpy().astype('uint8'), mode='L')
black_img.save("black_image.jpg")