# BioASQ 2020 Challenge

This repository contains our team's solution for the BioASQ 2020 challenge, a large-scale biomedical semantic indexing and question answering competition.

## Challenge Overview

The BioASQ challenge focuses on biomedical semantic indexing and question answering. It aims to push the frontiers of large-scale biomedical semantic indexing and question answering systems. The challenge consists of various tasks, and our team participated in Task 8b, which involves biomedical question answering.

## Repository Structure

The repository is organized into several main directories:

- `data/`: Contains the dataset files used for training and testing.
- `task_factoid/`: Code and notebooks for the factoid question answering task.
- `task_list/`: Code and notebooks for the list question answering task.
- `task_yesno/`: Code and notebooks for the yes/no question answering task.
- `utils/`: Utility functions and helper scripts used across different tasks.
- `transformers_models/`: Pre-trained models used in the project.

## Tasks

We addressed three main types of questions in the BioASQ challenge:

1. **Factoid Questions**: These are questions that require a specific fact or short phrase as an answer.
2. **List Questions**: These questions expect a list of items as the answer.
3. **Yes/No Questions**: These are binary questions that can be answered with either "yes" or "no".

## Approach

Our approach to solving these tasks involved:

1. **Data Preprocessing**: We used various NLP techniques to clean and prepare the biomedical text data.
2. **Embeddings**: We utilized pre-trained biomedical embeddings, including BioBERT and ELMo, to represent the text data.
3. **Model Architecture**: We implemented deep learning models using TensorFlow and Keras, with BERT-based architectures for factoid and list questions, and a custom neural network for yes/no questions.
4. **Training**: We fine-tuned our models on the BioASQ dataset, using techniques like early stopping to prevent overfitting.
5. **Evaluation**: We used various metrics to evaluate our models, including strict accuracy, lenient accuracy, and mean reciprocal rank for factoid and list questions, and standard classification metrics for yes/no questions.

## Key Files

- `task_factoid/train_factoid.py`: Main script for training the factoid question answering model.
- `task_list/functions_list.py`: Contains functions for the list question answering task.
- `task_yesno/notebooks/simple_embeddings.ipynb`: Notebook demonstrating the use of embeddings for yes/no questions.
- `utils/data.py`: Utility functions for data loading and preprocessing.
- `utils/evaluate.py`: Functions for evaluating model performance.

## Requirements

The project requires several Python libraries, which are listed in the `requirements.txt` file. Key dependencies include:

- TensorFlow
- PyTorch
- Transformers
- Flair
- NLTK
- NumPy
- Pandas

## Usage

To run the models:

1. Install the required dependencies: `pip install -r requirements.txt`
2. Prepare the data by running the data preprocessing scripts in the `utils/` directory.
3. Train the models using the respective training scripts in each task directory.
4. Evaluate the models using the provided evaluation functions.

## Results

Our models achieved competitive results on the BioASQ 2020 challenge. Detailed performance metrics and analysis can be found in the PDF summary in the main folder.

## Acknowledgements

We would like to thank the BioASQ organizers for providing this challenging and important competition in the field of biomedical text mining and question answering.

