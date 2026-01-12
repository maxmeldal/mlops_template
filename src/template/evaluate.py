import os
import torch
import typer
from template.data import corrupt_mnist
from template.model import MyAwesomeModel
import wandb

api = wandb.Api()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(alias: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(alias)

    model = MyAwesomeModel().to(DEVICE)
    artifact = api.artifact(f"maxmeldal/model-registry/corrupt_mnist_models:{alias}", type="model")
    artifact_dir = artifact.download()

    model_path = os.path.join(artifact_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
