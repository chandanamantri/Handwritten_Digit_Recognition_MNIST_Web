# Handwritten Digit Recognition - MNIST Version

A local Flask web application that recognizes handwritten digits 0-9.

## Setup

Create/activate a Python 3.14 virtual environment, then:

    pip install -r requirements.txt

Train the model:

    python train_model.py

The first training run downloads MNIST from OpenML. Internet is required only for this dataset download.

Start the website:

    python app.py

Open http://127.0.0.1:5000

## Safety

The application runs locally. It does not require administrator privileges and does not modify unrelated files. The training step downloads the public MNIST dataset through scikit-learn/OpenML and saves the trained model inside this project folder.
