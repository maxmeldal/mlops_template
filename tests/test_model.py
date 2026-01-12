from template.data import corrupt_mnist
from template.model import MyAwesomeModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def test_model_output_shape():
    """Test that the model output has the correct shape."""
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    sample_img, _ = train_set[0]
    sample_img = sample_img.unsqueeze(0).to(DEVICE)  # Add batch dimension
    output = model(sample_img)
    assert output.shape == (1, 10)
