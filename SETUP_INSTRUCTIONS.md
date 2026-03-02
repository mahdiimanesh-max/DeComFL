# DeComFL Environment Setup Instructions

## Prerequisites

- Python >= 3.10
- `uv` package manager

## Step 1: Install `uv`

Since `uv` is not currently installed, please install it using one of these methods:

### Option 1: Official Install Script (Recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, add `uv` to your PATH:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

To make this permanent, add the export line to your `~/.zshrc` (or `~/.bashrc`):
```bash
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Option 2: Homebrew (macOS)
```bash
brew install uv
```

### Option 3: pip
```bash
pip install uv
```

## Step 2: Verify Installation

Check that `uv` is installed:
```bash
uv --version
```

## Step 3: Set Up the Environment

Navigate to the DeComFL directory and run:

```bash
cd /Users/mehdiiranmanesh/Desktop/DeComFL
uv sync --extra dev
```

This will:
- Create a virtual environment (`.venv/`)
- Install all project dependencies
- Install development dependencies (mypy, pytest, isort)

## Step 4: Verify Setup

Test the setup by running one of the example commands:

```bash
# Test with a simple command
uv run python -c "import torch; print('PyTorch version:', torch.__version__)"

# Or check the help for one of the main scripts
uv run python decomfl_main.py --help
```

## Using the Environment

### Method 1: Using `uv run` (Recommended)
Prefix any Python command with `uv run`:
```bash
uv run python decomfl_main.py --dataset=sst2 --num-clients=3
```

### Method 2: Activate Virtual Environment
```bash
source .venv/bin/activate  # macOS/Linux
# Then run commands normally
python decomfl_main.py --dataset=sst2 --num-clients=3
```

## Troubleshooting

### If `uv sync` fails:
1. Make sure you have Python >= 3.10 installed: `python3 --version`
2. Check your internet connection
3. Try updating `uv`: `uv self update`

### If you get permission errors:
- Make sure you have write permissions in the project directory
- Try running without `--extra dev` first: `uv sync`

### If dependencies fail to install:
- Check that you have the required system dependencies
- For PyTorch with CUDA, ensure you have the correct CUDA version

## Next Steps

Once setup is complete, you can:
1. Run experiments using the commands in the main README.md
2. Check out the example commands in the "Run Experiments" section
3. Start developing by following the code style guidelines in `dev_tools/README.md`
