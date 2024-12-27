from manim import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def generate_smile_data(num_points=1000):
    X = np.random.uniform(-1, 1, (num_points, 2))
    Y = np.zeros(num_points)  # Labels: 0 = outside, 1 = part of smiley

    for i, (x, y) in enumerate(X):
        if x**2 + y**2 < 1:  # Outer circle
            if not (0.05 > ((x + 0.35) ** 2 + (y - 0.35) ** 2)):  # Exclude left eye
                if not (
                    0.05 > ((x - 0.35) ** 2 + (y - 0.35) ** 2)
                ):  # Exclude right eye
                    if not (
                        0.5 < (x**2 + (y - 0.15) ** 2) < 0.7
                    ):  # Exclude smile curve
                        Y[i] = 1

    for i, (x, y) in enumerate(X):
        if 0.6 < x**2 + y**2 < 1:
            if 0.01 < y < 1:
                Y[i] = 1

    for i, (x, y) in enumerate(X):
        if x**2 + y**2 < 1:  # Outer circle
            if 0.05 > ((x + 0.35) ** 2 + (y - 0.35) ** 2):  # Exclude left eye
                if 0.05 > ((x - 0.35) ** 2 + (y - 0.35) ** 2):  # Exclude right eye
                    Y[i] = 0

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


class SmileFittingModel(nn.Module):
    def __init__(self, input_shape=2, hidden_shape=256, output_shape=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.LeakyReLU(),
            nn.Linear(hidden_shape, output_shape),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class SmileyLearningScene(Scene):
    def construct(self):
        x, y = generate_smile_data(num_points=5000)
        y = y.unsqueeze(dim=1)
        model = SmileFittingModel()
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        grid_size = 50
        x_test = np.linspace(-1, 1, grid_size)
        y_test = np.linspace(-1, 1, grid_size)
        grid = np.array([[xi, yi] for xi in x_test for yi in y_test])
        grid_tensor = torch.tensor(grid, dtype=torch.float32)

        # Create heatmap squares (static objects)
        heatmap_squares = self.create_heatmap(grid_size)
        self.play(FadeIn(heatmap_squares), run_time=0.05)

        # Training loop
        epochs = 250
        for epoch in range(epochs):
            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
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
                square.move_to(np.array([i * square_size - 1, j * square_size - 1, 0]))
                heatmap_squares.add(square)

        heatmap_squares.move_to(ORIGIN)
        return heatmap_squares

    def update_heatmap_colors(self, heatmap_squares, data):
        min_value, max_value = np.min(data), np.max(data)
        norm_data = (data - min_value) / (max_value - min_value)

        color_gradient = [BLACK, DARK_GREY, GRAY, LIGHT_GREY, WHITE]

        for idx, square in enumerate(heatmap_squares):
            i = idx // 50  # Row index
            j = idx % 50  # Column index
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
