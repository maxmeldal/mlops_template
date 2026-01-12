from torch.utils.data import Dataset

from template.data import corrupt_mnist


def test_my_dataset():
    """Test the MyDataset class."""
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    assert train[0][0].shape == (1, 28, 28)
    assert test[0][0].shape == (1, 28, 28)
    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)