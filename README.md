# Lebesgue D/R₃/R₅ Test-Body Demo

Interactive visualization of the finite Reuleaux test-body method for deriving a lower bound in **Lebesgue's universal covering problem**.

The app places three diameter-one test bodies — the disk **D**, the Reuleaux triangle **R₃**, and the Reuleaux pentagon **R₅** — in a common convex hull. Move the bodies interactively to see how the sampled convex-hull area changes.

## Live Demo

[https://lebesgue-5param-demo.streamlit.app](https://lebesgue-5param-demo.streamlit.app)

## Quick Start

```bash
# Install uv if not already installed
# https://docs.astral.sh/uv/

# Create virtual environment and install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run lebesgue_5_param_demo.py
```

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (or manually create a venv and `pip install .`)

## Controls

| Parameter | Description |
|-----------|-------------|
| x₃, y₃ | Horizontal / vertical shift of the Reuleaux triangle R₃ |
| ψ₅ | Rotation of the Reuleaux pentagon R₅ (degrees) |
| x₅, y₅ | Horizontal / vertical shift of the Reuleaux pentagon R₅ |

Use the sidebar to adjust parameters via sliders or exact numeric input. The sampled area of the convex hull is shown in real time.


