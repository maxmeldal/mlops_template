import torch

from template.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def test_model_output_shape():
    """Test that the model output has the correct shape."""
    model = MyAwesomeModel().to(DEVICE)

    # Create synthetic input data
    sample_img = torch.randn(1, 1, 28, 28).to(DEVICE)

    output = model(sample_img)
    assert output.shape == (1, 10)


def test_model_batch_processing():
    """Test that the model can handle different batch sizes."""
    model = MyAwesomeModel().to(DEVICE)

    for batch_size in [1, 8, 32]:
        sample_batch = torch.randn(batch_size, 1, 28, 28).to(DEVICE)
        output = model(sample_batch)
        assert output.shape == (batch_size, 10)


def test_model_output_range():
    """Test that model outputs are logits (unbounded)."""
    model = MyAwesomeModel().to(DEVICE)
    sample_img = torch.randn(5, 1, 28, 28).to(DEVICE)

    output = model(sample_img)
    # Logits should be unbounded, check they're not all zeros
    assert output.abs().sum() > 0
