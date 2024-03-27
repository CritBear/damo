import os
from pathlib import Path


class Paths:
    base = Path(__file__).resolve().parent.parent.parent
    support_data = base / 'support_data'
    datasets = base / 'datasets'
    trained_models = base / 'trained_models'
    modules = base / 'modules'
    test_results = base / 'test_results'

