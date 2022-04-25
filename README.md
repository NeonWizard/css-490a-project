# CSS 490a Acceleration Project
Aiming to accelerate neural network inference time on embedded systems via model compression, specifically pruning and quantization.

## Setup
```bash
virtualenv --python $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# classify image with unmodified VGG-16
python run_classify.py

# prune model
python prune_model.py
```