def pytest_addoption(parser):
    parser.addoption("--config", action="store", type=str)
    parser.addoption('--patient', action='store', type=int, default=98)
    parser.addoption('--save-path', action='store', type=str, default='examples')
    parser.addoption('--splits', action='store', type=str, default='val')
    parser.addoption("--n-images", action="store", type=int, default=10)
