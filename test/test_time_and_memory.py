import os
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(f"{os.getcwd()}")  # Should be the JAFAR repo.
from src.upsampler import JAFAR


@pytest.fixture(params=["JAFAR"])
def model_fixture(request):
    dim = 384
    lr_size = 16
    if request.param == "JAFAR":
        model = JAFAR(input_dim=3, qk_dim=128, v_dim=dim).cuda()
    return request.param, model


@pytest.fixture(params=[56, 112, 224, 448])
def setup_data(request):
    dim = 384
    img_size = request.param
    batch_size = 1

    lr_feats = torch.randn(batch_size, dim, img_size // 14, img_size // 14).cuda()
    img = torch.randn(batch_size, 3, img_size, img_size).cuda()

    return lr_feats, img, img_size


def test_forward_speed(model_fixture, setup_data):
    model_name, model = model_fixture
    lr_feats, img, img_size = setup_data

    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            _ = model(img, lr_feats, (img_size, img_size))

    num_runs = 10
    total_time = 0

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        torch.cuda.synchronize()  # Ensure all previous operations are complete
        start_event.record()
        with torch.no_grad():
            _ = model(img, lr_feats, (img_size, img_size))
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete
        total_time += start_event.elapsed_time(end_event)  # Time in milliseconds

    avg_time = total_time / num_runs
    print(f"{model_name} - Resolution {img_size}x{img_size} - Average forward pass time: {avg_time:.6f} ms")


def test_backward_speed(model_fixture, setup_data):
    model_name, model = model_fixture
    lr_feats, img, img_size = setup_data
    linear = nn.Conv2d(384, 1, 1).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(linear.parameters()), lr=0.01)

    # Warmup runs
    for _ in range(5):
        lr_feats.requires_grad = True
        img.requires_grad = True
        output = model(img, lr_feats, (img_size, img_size))
        loss = linear(output).sum()
        loss.backward()
        optimizer.step()

    num_runs = 10
    total_time = 0

    # CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        lr_feats.requires_grad = True
        img.requires_grad = True
        torch.cuda.synchronize()  # Ensure all previous operations are complete
        start_event.record()
        output = model(img, lr_feats, (img_size, img_size))
        loss = linear(output).sum()
        loss.backward()
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to complete
        total_time += start_event.elapsed_time(end_event)  # Time in milliseconds

    avg_time = total_time / num_runs
    print(f"{model_name} - Resolution {img_size}x{img_size} - Average backward pass time: {avg_time:.6f} ms")


def test_gpu_memory_usage_forward(model_fixture, setup_data):
    torch.cuda.empty_cache()
    model_name, model = model_fixture
    model.eval()
    lr_feats, img, img_size = setup_data

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(img, lr_feats, (img_size, img_size))
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to M

    print(f"{model_name} - Resolution {img_size}x{img_size} - Peak GPU memory usage (forward): {peak_memory:.2f} MB")


def test_gpu_memory_usage_backward(model_fixture, setup_data):
    torch.cuda.empty_cache()
    model_name, model = model_fixture
    model.train()
    lr_feats, img, img_size = setup_data
    linear = nn.Conv2d(384, 1, 1).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(linear.parameters()), lr=0.01)

    # Measure GPU memory usage during backward pass
    torch.cuda.reset_peak_memory_stats()
    lr_feats.requires_grad = True
    img.requires_grad = True
    output = model(img, lr_feats, (img_size, img_size))
    loss = linear(output).sum()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB

    print(f"{model_name} - Resolution {img_size}x{img_size} - Peak GPU memory usage (backward): {peak_memory:.2f} MB")
