import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform, phase='train'):
        # print(root_dir)
    
        self.input_dir = os.path.join("/content/UW/Projects",root_dir, phase, 'input')
        self.target_dir = os.path.join("/content/UW/Projects",root_dir, phase, 'target')
        self.transform = transform
        # print(f"Input directory: {self.input_dir}")
        # print(f"Target directory: {self.target_dir}")

        # Ensure input and target directories exist
        if not os.path.exists(self.input_dir) or not os.path.exists(self.target_dir):
            raise FileNotFoundError(f"Directories {self.input_dir} or {self.target_dir} not found.")

        # List all image filenames (assuming they match in both input and target directories)
        self.image_filenames = sorted(os.listdir(self.input_dir))
        print(self.input_dir,self.target_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Construct file paths
        input_path = os.path.join(self.input_dir, self.image_filenames[idx])
        target_path = os.path.join(self.target_dir, self.image_filenames[idx])

        # Load images
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image