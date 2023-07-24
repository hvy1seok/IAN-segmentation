import pytest

from visualize import visualize_data, save_predictions


def test_save_predictions(request):
    config = request.config.getoption('--config')
    n_images = request.config.getoption('--n-images')
    save_dir = request.config.getoption('--save-dir')
    splits = request.config.getoption('--splits')
    
    save_predictions(config, save_dir, n_images=n_images, splits=splits)


def test_visualize(request):
    patient = request.config.getoption('--patient')
    save_dir = request.config.getoption('--save-dir')
    splits = request.config.getoption('--splits')

    visualize_data(patient, save_dir, splits)
