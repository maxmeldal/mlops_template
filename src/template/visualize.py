
import matplotlib.pyplot as plt
import torch
import typer
from template.model import MyAwesomeModel
from template.data import corrupt_mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc = torch.nn.Identity()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    embeddings, targets = [], []
    with torch.inference_mode():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            predictions = model(img)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).to("cpu").numpy()
        targets = torch.cat(targets).to("cpu").numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)