import torchvision
import os
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def download_mnist(data_dir: str = "data/raw"):
    """Download MNIST dataset to specified directory."""
    try:
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("Downloading MNIST dataset...")
        torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True
        )
        torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True
        )
        logger.info("Dataset downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_mnist()
