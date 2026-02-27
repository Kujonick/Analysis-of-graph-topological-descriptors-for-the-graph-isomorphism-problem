# Analysis of graph topological descriptors for the graph isomorphism problem
This Repo is a codebase for my Master's Thesis (of the same title) and continuation of my Pre-Thesis Seminar, which were then done with [Michał](https://github.com/mnozkiewicz)

## Setup

To setup the project, you will need [uv](https://github.com/astral-sh/uv). You can download it via `pip` or `pipx`. After instalation you can run:

```
uv sync
source .venv/bin/activate
python main.py
```

### Precommit 

To enable automatic pre-commit hooks on every commit, run:
```
pre-commit install
```

To manually run all configured hooks on (without making a commit), use:
```
pre-commit run --all-files
```

