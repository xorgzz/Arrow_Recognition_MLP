import torch
import torch.nn as nn


def extract_features(matrix):
    if torch.is_tensor(matrix):
        matrix = matrix.reshape(8, 8).tolist()
    elif isinstance(matrix, list) and len(matrix) == 64 and not isinstance(matrix[0], list):
        matrix = [matrix[i:i + 8] for i in range(0, 64, 8)]

    proj_x = [0.0] * 8
    proj_y = [0.0] * 8

    for r in range(8):
        for c in range(8):
            if matrix[r][c] == 1:
                proj_y[r] += 1.0
                proj_x[c] += 1.0


    return proj_x + proj_y


class ShapeClassifier(nn.Module):
    def __init__(self):
        super(ShapeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 13),
            nn.ReLU(),
            nn.Linear(13, 4)
        )

    def forward(self, x):
        return self.network(x)


def load_model(path="model_strzalek.pth"):
    model = ShapeClassifier()
    try:
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
    except:
        return None


def predict(model, matrix):
    features = extract_features(matrix)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        return int(torch.max(output, 1)[1].item())