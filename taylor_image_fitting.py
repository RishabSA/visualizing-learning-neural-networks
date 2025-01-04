from manim import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_image_data(image_path: str):
    image = Image.open(image_path).convert("L").resize((128, 128))
    image_array = np.array(image) / 255.0
    height, width = image_array.shape
    X, Y, intensities = [], [], []
    for y in range(height):
        for x in range(width):
            X.append(x / width)
            Y.append(1 - y / height)
            intensities.append(1.0 - image_array[y, x])  # Intensity value (grayscale)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    intensities = torch.tensor(intensities, dtype=torch.float32)

    return torch.cat([X, Y], dim=1), intensities


class NormalizedTanh(nn.Module):
    def forward(self, x):
        return (torch.tanh(x) + 1) / 2


class NNFittingModel(nn.Module):
    def __init__(self, input_shape=1, hidden_shape=128, output_shape=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, output_shape),
            NormalizedTanh(),
        )

    def forward(self, x):
        return self.layers(x)


class TaylorImageFittingModel(nn.Module):
    def __init__(self, taylor_order=4, input_shape=2, hidden_shape=128, output_shape=1):
        super().__init__()
        self.taylor_order = taylor_order
        input_shape = input_shape * taylor_order
        self.inner_model = NNFittingModel(
            input_shape=input_shape,
            hidden_shape=hidden_shape,
            output_shape=output_shape,
        )
        self.orders = torch.arange(1, taylor_order + 1).float()

    def forward(self, x):
        x = x.unsqueeze(-1)
        taylor_features = torch.pow(x, self.orders)
        taylor_features = taylor_features.view(x.shape[0], -1)
        return self.inner_model(taylor_features)


class TaylorImageLearningScene(Scene):
    def construct(self):
        x, y = get_image_data("images/curly_hair.png")
        y = y.unsqueeze(dim=1)

        model = TaylorImageFittingModel(taylor_order=16)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        grid_size = 128
        x_test = np.linspace(0, 1, grid_size)
        y_test = np.linspace(0, 1, grid_size)
        grid = np.array([[xi, yi] for xi in x_test for yi in y_test])
        grid_tensor = torch.tensor(grid, dtype=torch.float32)

        # Create heatmap squares (static objects)
        heatmap_squares = self.create_heatmap(grid_size)
        self.play(FadeIn(heatmap_squares), run_time=0.05)

        # Training loop
        epochs = 1000
        for epoch in range(epochs):
            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 200 == 0:
                model.eval()
                with torch.inference_mode():
                    predictions = (
                        model(grid_tensor).numpy().reshape(grid_size, grid_size)
                    )

                self.update_heatmap_colors(heatmap_squares, predictions)

                self.wait(0.05)

        self.wait()

    def create_heatmap(self, grid_size):
        square_size = 5 / (grid_size)
        heatmap_squares = VGroup()

        for i in range(grid_size):
            for j in range(grid_size):
                square = Square(
                    side_length=square_size, color=WHITE, fill_opacity=1, stroke_width=0
                )
                square.move_to(
                    np.array([i * square_size - 2.5, j * square_size - 2.5, 0])
                )
                heatmap_squares.add(square)

        heatmap_squares.move_to(ORIGIN)
        return heatmap_squares

    def update_heatmap_colors(self, heatmap_squares, data):
        min_value, max_value = np.min(data), np.max(data)
        norm_data = (data - min_value) / (max_value - min_value)

        color_gradient = [BLACK, DARK_GREY, GRAY, LIGHT_GREY, WHITE]

        for idx, square in enumerate(heatmap_squares):
            i = idx // 128  # Row index
            j = idx % 128  # Column index
            value = norm_data[i, j]
            color = self.get_color_from_value(value, color_gradient)
            square.set_fill(color, opacity=1)

    def get_color_from_value(self, value, color_gradient):
        if value < 0.25:
            return interpolate_color(color_gradient[0], color_gradient[1], value * 4)
        elif value < 0.5:
            return interpolate_color(
                color_gradient[1], color_gradient[2], (value - 0.25) * 4
            )
        elif value < 0.75:
            return interpolate_color(
                color_gradient[2], color_gradient[3], (value - 0.5) * 4
            )
        else:
            return interpolate_color(
                color_gradient[3], color_gradient[4], (value - 0.75) * 4
            )
