from manim import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)

x = torch.linspace(-10, 10, 200).reshape(-1, 1)
y = torch.sin(2 * x) - torch.cos(3 * x)


class CurveFittingModel(nn.Module):
    def __init__(self, input_shape=1, hidden_shape=128, output_shape=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape, output_shape),
        )

    def forward(self, x):
        return self.layers(x)


class FourierCurveFittingModel(nn.Module):
    def __init__(
        self, fourier_order=16, input_shape=1, hidden_shape=128, output_shape=1
    ):
        super().__init__()
        self.fourier_order = fourier_order
        input_shape = input_shape * 2 * fourier_order + input_shape
        self.inner_model = CurveFittingModel(
            input_shape=input_shape,
            hidden_shape=hidden_shape,
            output_shape=output_shape,
        )
        self.orders = torch.arange(1, fourier_order + 1).float()

    def forward(self, x):
        x = x.unsqueeze(-1)
        fourier_features = torch.cat(
            [torch.sin(self.orders * x), torch.cos(self.orders * x), x], dim=-1
        )
        fourier_features = fourier_features.view(x.shape[0], -1)
        return self.inner_model(fourier_features)


class FourierCurveFittingAnimation(Scene):
    def construct(self):
        model = FourierCurveFittingModel(fourier_order=16)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)

        axes = Axes(
            x_range=[-10, 10, 1],
            y_range=[-4, 4, 1],
            axis_config={"color": WHITE},
        )

        scatter_points = VGroup(
            *[
                Dot(axes.c2p(xi.item(), yi.item()), color=BLUE, radius=0.05)
                for xi, yi in zip(x, y)
            ]
        )

        x_vals = x.detach().numpy()
        initial_preds = np.zeros_like(x_vals)
        line_graph = axes.plot_line_graph(
            x_vals.flatten(),
            initial_preds.flatten(),
            add_vertex_dots=False,
            line_color=RED,
        )

        # Add elements to the scene
        self.add(axes, scatter_points, line_graph)

        # Training Loop
        epochs = 500
        for epoch in range(epochs):
            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                model.eval()
                with torch.no_grad():
                    updated_preds = model(x).detach().numpy()
                new_graph = axes.plot_line_graph(
                    x_vals.flatten(),
                    updated_preds.flatten(),
                    add_vertex_dots=False,
                    line_color=RED,
                )
                self.play(Transform(line_graph, new_graph), run_time=0.02)

        # Final Display
        self.wait()
