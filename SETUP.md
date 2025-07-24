# MLOps Course - Quick Start Guide

## Installation

This project uses `uv` for fast Python package management. To get started:

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Install all dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
uv shell
```

## Dependencies Included

### Data Science Libraries
- **NumPy, Pandas, SciPy**: Core data manipulation and scientific computing
- **Scikit-learn, XGBoost, LightGBM**: Machine learning algorithms
- **PyTorch, TensorFlow**: Deep learning frameworks
- **Matplotlib, Seaborn, Plotly, Bokeh**: Data visualization
- **Polars, Dask**: High-performance data processing

### MLOps Tools
- **MLflow**: Experiment tracking and model management
- **Wandb**: Experiment tracking and collaboration
- **Optuna**: Hyperparameter optimization

### Web API
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **Pydantic**: Data validation and serialization

### Tunneling
- **pyngrok**: Python wrapper for ngrok to expose local servers

### Development Tools
- **Jupyter/JupyterLab**: Interactive development environment
- **Black, isort, flake8**: Code formatting and linting
- **pytest**: Testing framework

## Quick Start Examples

### Run FastAPI server with ngrok tunnel:
```python
# See examples/fastapi_with_ngrok.py
uv run python examples/fastapi_with_ngrok.py
```

### Start Jupyter Lab:
```bash
uv run jupyter lab
```

### Run ML experiment:
```python
# See examples/ml_experiment.py
uv run python examples/ml_experiment.py
```
