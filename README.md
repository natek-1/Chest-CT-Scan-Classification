# Chess Disease Classification

This project focuses on classifying chess disease images into two categories: normal and Adenocarcinoma. Utilizing advanced machine learning techniques, we aim to provide accurate and efficient diagnosis support.

## Table of Contents

- [Introduction](#introduction)
- [Link to Dataset](#Dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Chess Disease Classification project leverages deep learning models to differentiate between normal and Adenocarcinoma chess images. This application is designed to aid medical professionals by providing a reliable automated classification tool. By integrating various tools and technologies, we ensure an end-to-end solution from data handling to model deployment.

## Dataset

Here is the link to the [Dataset](https://drive.google.com/file/d/1z0mreUtRmR-P-magILsDR3T7M6IkGXtY/view).


## Project Structure

```
chess-disease-classification/
│
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed data files
│
├── notebooks/               # Jupyter notebooks for exploration and prototyping
│
├── models/                  # Trained models
│
├── src/                     # Source code for model training and evaluation
│   ├── data_processing.py   # Data preprocessing scripts
│   ├── model.py             # Model architecture and training
│   ├── evaluation.py        # Evaluation metrics and visualization
│
├── app/                     # Flask application for deployment
│   ├── app.py               # Flask API
│
├── Jenkinsfile              # Jenkins pipeline configuration
│
└── README.md                # Project documentation
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chess-disease-classification.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data version control setup:**
   ```bash
   dvc init
   dvc remote add -d myremote <remote-storage-url>
   ```

4. **Pull the dataset:**
   ```bash
   dvc pull
   ```

## Usage

1. **Train the model:**
   ```bash
   python src/model.py
   ```

2. **Evaluate the model:**
   ```bash
   python src/evaluation.py
   ```

3. **Run the Flask application:**
   ```bash
   python app/app.py
   ```

4. **Access the app at:** `http://localhost:5000`

## Technologies Used

- **TensorFlow:** For model development and training.
- **MLflow:** To track and manage machine learning experiments.
- **DVC:** For data version control and management.
- **Jenkins:** For continuous integration and deployment.
- **Flask:** For building the web application.
- **Docker:** dockerize the application to facilitate deployment.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
