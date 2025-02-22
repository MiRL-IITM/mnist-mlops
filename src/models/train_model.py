import torch
import mlflow
from .model import MNISTModel
from ..data.preprocess_data import preprocess_data
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def train(epochs: int = 10, lr: float = 0.01):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        train_loader, _ = preprocess_data()
        
        with mlflow.start_run():
            mlflow.log_param("lr", lr)
            mlflow.log_param("epochs", epochs)
            
            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"Train Epoch: {epoch} "
                                  f"Loss: {loss.item():.6f}")
                        mlflow.log_metric("loss", loss.item())
                        
        torch.save(model.state_dict(), "checkpoints/trained_model.pth")
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train()
