# Music Genre Classification

ML proj for classifying music genres using audio features.

## Setup:

### Pre-reqs
- Python 3.12 (required for TensorFlow)
- Git

### Using pip

```bash
# clone the repository
git clone https://github.com/rpc1001/music-genre-ml.git
cd music-genre-ml

# create a virtual environment with Python 3.12
python3.12 -m venv venv

# activate the virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# register the venv as a Jupyter kernel
python -m ipykernel install --user --name=venv --display-name="Python (venv 3.12.2)"

# Launch Jupyter notebook
jupyter notebook
```

### Run the frontend

```bash
# make sure models are trained then start the Flask app
python app.py

# open browser to http://localhost:8080
```