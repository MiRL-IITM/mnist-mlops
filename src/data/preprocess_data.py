from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def preprocess_data(raw_data_dir: str = "data/raw", 
                   processed_data_dir: str = "data/processed",
                   batch_size: int = 32):
    """Preprocess MNIST data and create data loaders."""
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load training data
        train_dataset = MNIST(raw_data_dir, train=True, transform=transform)
        
        # Split into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        print(train_size, val_size)
        print(train_size//32)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info("Data preprocessing completed successfully")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()
