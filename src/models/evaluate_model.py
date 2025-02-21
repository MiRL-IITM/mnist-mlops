import torch
import mlflow
from .model import MNISTModel
from ..data.preprocess_data import preprocess_data
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

def evaluate(model_path: str = "checkpoints/trained_model.pth"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MNISTModel().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        _, val_loader = preprocess_data()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        logger.info(f'Validation accuracy: {accuracy}%')
        
        with mlflow.start_run():
            mlflow.log_metric("val_accuracy", accuracy)
            
        return accuracy
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate()
