from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PairedImageDataset
import yaml

def get_dataloader(config_path):

    # Loading config file ->Done successfully
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #Loading configurations -> done successfully
    dataset_path = config['data']['dataset_path']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    # Transfoming the images -> done successfully
    tran = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    # Importing data -> ?
    dataset = PairedImageDataset(dataset_path, transform=tran)

    # Create the DataLoader -> ?
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
