# MNIST MLOps Tutorial

This project demonstrates MLOps best practices using the MNIST dataset.

## Project Structure

- `data/`: Raw and processed data
- `src/`: Source code
  - `data/`: Data processing scripts
  - `models/`: Model definition and training
  - `api/`: REST API service
  - `utils/`: Utility functions
  - `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration
- `Dockerfile`: Container definition
- `docker-compose.yml`: Multi-container setup
- `Makefile`: Build automation
- `requirements.txt`: Python dependencies
- `dvc.yaml`: Data version control

## Setup

```bash
make setup
```

## Usage

```bash
make train
make serve
```
