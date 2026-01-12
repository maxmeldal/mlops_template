import torch
from torch.utils.data import Dataset, TensorDataset
import pytest
import os

from template.data import normalize


@pytest.fixture
def setup_test_data(tmp_path, monkeypatch):
    """Create temporary mock data files for testing."""
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    
    # Create synthetic data
    train_images = torch.randn(30000, 1, 28, 28)
    train_target = torch.randint(0, 10, (30000,))
    test_images = torch.randn(5000, 1, 28, 28)
    test_target = torch.randint(0, 10, (5000,))
    
    # Save to temp directory
    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_target, processed_dir / "train_target.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_target, processed_dir / "test_target.pt")
    
    # Change to temp directory so relative paths work
    monkeypatch.chdir(tmp_path)
    
    return tmp_path


def test_my_dataset(setup_test_data):
    """Test the MyDataset class."""
    from template.data import corrupt_mnist
    
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    assert train[0][0].shape == (1, 28, 28)
    assert test[0][0].shape == (1, 28, 28)
    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)


def test_normalize():
    """Test the normalize function."""
    images = torch.randn(100, 1, 28, 28)
    normalized = normalize(images)
    
    # Check that mean is close to 0 and std is close to 1
    assert torch.abs(normalized.mean()) < 0.01
    assert torch.abs(normalized.std() - 1.0) < 0.01
