import logging
import os

import hydra
import torch
import wandb
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision.utils import make_grid

from template.data import corrupt_mnist
from template.model import MyAwesomeModel

load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg) -> None:
    run = wandb.init(entity="maxmeldal", project="mlops", config=cfg.hyperparameters)
    """Train a model on MNIST."""
    logging.info("Training day and night")
    lr = cfg.hyperparameters.learning_rate
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.num_epochs
    logging.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        preds_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds_list.append(y_pred.detach().cpu())
            targets_list.append(target.detach().cpu())

            if i % 100 == 0:
                logging.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a grid of the input images
                grid = make_grid(img[:25].detach().cpu(), nrow=5, normalize=True)  # [3, H, W] or [1, H, W]
                wandb.log({"image_grid": wandb.Image(grid)})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0).cpu()
                wandb.log({"gradients": wandb.Histogram(grads.numpy().tolist())})

        preds: torch.Tensor = torch.cat(preds_list, 0)
        targets: torch.Tensor = torch.cat(targets_list, 0)

    logging.info("Training complete")

    # make sure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    torch.save(model.state_dict(), "models/model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)

    run.link_artifact(artifact=artifact, target_path="model-registry/corrupt_mnist_models", aliases=["latest"])


if __name__ == "__main__":
    train()
