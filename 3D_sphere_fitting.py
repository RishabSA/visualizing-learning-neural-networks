from manim import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

torch.manual_seed(0)


class SphereFittingModel(nn.Module):
    def __init__(self, input_shape=2, hidden_shape=50, output_shape=3):
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
        )

    def forward(self, x):
        return self.layers(x)


def shift_value(x, start_range, end_range):
    return (x - start_range[0]) / (start_range[1] - start_range[0]) * (
        end_range[1] - end_range[0]
    ) + end_range[0]


def generate_inputs(u_data, v_data, u_range, v_range, nn_range=None):
    if nn_range is not None:
        u_data = shift_value(u_data, u_range, nn_range)
        v_data = shift_value(v_data, v_range, nn_range)
    inputs = torch.Tensor(np.array([u_data, v_data]).T)
    return inputs


def generate_outputs(x, y, z):
    return torch.Tensor(np.array([x, y, z]).T)


def generate_display_points(x_data, y_data, z_data, num_display_samples):
    data_points = [np.array([x, y, z]) for x, y, z in zip(x_data, y_data, z_data)]
    return random.sample(data_points, num_display_samples)


def precomute_net_outputs(scene, net, u_range, v_range, resolution):
    """
    Takes in u, v and returns the output of the network
    """

    inputs = []

    def dummy_func(u, v):
        inputs.append([u, v])
        return np.array([0.0, 0.0, 0.0])

    dummy_surface = Surface(
        dummy_func,
        resolution=resolution,
        u_range=u_range,
        v_range=v_range,
        checkerboard_colors=[BLUE_D, BLUE_E],
    ).set_opacity(0)

    scene.add(dummy_surface)
    scene.remove(dummy_surface)

    net_inputs = torch.Tensor(inputs)
    outputs = net(net_inputs).detach().numpy()

    input_output_map = {}
    for i in range(len(inputs)):
        input_output_map[(inputs[i][0].item(), inputs[i][1].item())] = outputs[i]

    def approx_surface_func(u, v):
        return input_output_map[(u, v)]

    return approx_surface_func


def create_sphere_surface(
    scene,
    model,
    inputs,
    outputs,
    nn_range,
    resolution,
    epochs=10,
    batch_size=20,
):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_samples = len(inputs)

    approx_surface_func = precomute_net_outputs(
        scene, model, nn_range, nn_range, resolution
    )

    approx_surface = Surface(
        approx_surface_func,
        resolution=resolution,
        u_range=nn_range,
        v_range=nn_range,
        checkerboard_colors=[BLUE_D, BLUE_E],
    ).set_opacity(0.9)

    scene.add(approx_surface)

    for epoch in range(epochs):
        index_batches = np.array_split(np.random.permutation(num_samples), batch_size)
        total_loss = 0
        for i in index_batches:
            ins = inputs[i]
            outs = outputs[i]

            y_pred = model(ins)
            loss = loss_fn(y_pred, outs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch} | Loss: {(total_loss / num_samples):.10f}")
        approx_surface_func = precomute_net_outputs(
            scene, model, nn_range, nn_range, resolution
        )

        new_approx_surface = Surface(
            approx_surface_func,
            resolution=resolution,
            u_range=nn_range,
            v_range=nn_range,
            checkerboard_colors=[BLUE_D, BLUE_E],
        ).set_opacity(0.9)
        scene.play(
            ReplacementTransform(approx_surface, new_approx_surface),
            run_time=0.3,
            rate_func=linear,
        )
        approx_surface = new_approx_surface
        scene.add(approx_surface)

    return approx_surface


class SphereFittingScene(ThreeDScene):
    def construct(self):
        batch_size = 20
        num_samples = 1000
        num_display_samples = 200
        nn_range = [-PI, PI]

        size = 3
        resolution = (40, 40)
        u_range = [-PI / 2, PI / 2]
        v_range = [0, TAU]

        def sphere(u, v):
            return np.array(
                [
                    size * np.cos(u) * np.cos(v),
                    size * np.cos(u) * np.sin(v),
                    size * np.sin(u),
                ]
            )

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=-0.1)

        u_data = np.arccos(2 * np.random.uniform(0, 1, num_samples) - 1) - np.pi / 2
        v_data = np.random.uniform(0, 2 * np.pi, num_samples)
        x_data, y_data, z_data = sphere(u_data, v_data)

        display_points = generate_display_points(
            x_data, y_data, z_data, num_display_samples
        )
        inputs = generate_inputs(u_data, v_data, u_range, v_range, nn_range)
        outputs = generate_outputs(x_data, y_data, z_data)

        dots = [
            Dot3D(point=d, color=RED, radius=0.05, resolution=[5, 5])
            for d in display_points
        ]
        self.play(*[FadeIn(d) for d in dots])
        self.wait(1)

        model = SphereFittingModel(
            input_shape=2,
            hidden_shape=50,
            output_shape=3,
        )

        approx_surface = create_sphere_surface(
            scene=self,
            model=model,
            inputs=inputs,
            outputs=outputs,
            nn_range=nn_range,
            resolution=resolution,
            epochs=25,
            batch_size=batch_size,
        )
