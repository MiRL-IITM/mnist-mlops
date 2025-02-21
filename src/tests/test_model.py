import unittest
import torch
from ..models.model import MNISTModel

class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTModel()
        self.batch_size = 4
        self.input_shape = (self.batch_size, 1, 28, 28)
        
    def test_model_output_shape(self):
        x = torch.randn(self.input_shape)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 10))
        
    def test_model_forward_pass(self):
        x = torch.randn(self.input_shape)
        try:
            output = self.model(x)
            self.assertIsNotNone(output)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {str(e)}")
            
    def test_model_save_load(self):
        # Save model
        torch.save(self.model.state_dict(), "checkpoints/test_model.pth")
        
        # Load model
        loaded_model = MNISTModel()
        loaded_model.load_state_dict(torch.load("checkpoints/test_model.pth", weights_only=True))
        
        # Compare outputs
        x = torch.randn(self.input_shape)
        with torch.no_grad():
            output1 = self.model(x)
            output2 = loaded_model(x)
            
        self.assertFalse(torch.allclose(output1, output2))

if __name__ == '__main__':
    unittest.main()
