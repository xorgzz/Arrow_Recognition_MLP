import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gen_lib import gen_data_set
from mlp_utils import ShapeClassifier, extract_features


def main():
    print("--- UCZENIE MODELU (16 -> 13 -> 4) ---")

    patterns, labels = gen_data_set()

    print("Ekstrakcja 16 cech (Projekcji) z każdego obrazka...")
    features = [extract_features(p) for p in patterns]

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ShapeClassifier()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 400
    print(f"Rozpoczynanie głębokiego uczenia na {len(x)} próbkach przez {epochs} epok...")

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (torch.max(model(x), 1)[1] == y).sum().item() / y.size(0)
                print(f"Epoka {epoch + 1:4} | Celność treningowa: {acc * 100:.2f}%")

    torch.save(model.state_dict(), "model_strzalek.pth")
    print("Model zapisany!")


if __name__ == '__main__':
    main()