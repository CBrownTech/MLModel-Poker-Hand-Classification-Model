# Poker Hand Classification Model

This project implements a machine learning model to classify poker hands. It uses a `RandomForestClassifier` from `scikit-learn` to predict the type of a five-card poker hand, ranging from "Nothing" to "Royal flush".

The script is designed to train on the Poker Hand dataset from the UCI Machine Learning Repository, evaluate the model's performance, and provide a simple interface for predicting the category of a new, user-provided hand.

## Features

-   **Data Preprocessing**: Converts card representations (e.g., 'AS' for Ace of Spades) into numerical features suitable for machine learning.
-   **Model Training**: Trains a Random Forest Classifier on a labeled dataset of poker hands.
-   **Model Evaluation**: Assesses the model's performance on a test set using accuracy and a detailed classification report.
-   **Prediction**: Allows for the classification of new, unseen poker hands.

## Dataset

This model is trained and tested using the **Poker Hand Data Set** from the UCI Machine Learning Repository. The script expects the data to be split into two files:

-   `poker-hand-training.csv`
-   `poker-hand-testing.csv`

These files should be placed in a directory named `Poker Hand Analysis` inside your user's `Documents` folder (e.g., `C:\Users\YourUser\Documents\Poker Hand Analysis`).

The dataset format is a CSV file where each row represents a hand, with columns for the suit and rank of each of the five cards, followed by the hand's classification code.

-   **Hand Categories (Labels)**:
    -   0: Nothing in hand
    -   1: One pair
    -   2: Two pairs
    -   3: Three of a kind
    -   4: Straight
    -   5: Flush
    -   6: Full house
    -   7: Four of a kind
    -   8: Straight flush
    -   9: Royal flush

## Requirements

The project requires the following Python libraries:

-   `numpy`
-   `pandas`
-   `scikit-learn`

## Installation

1.  **Clone or download the project files.** Ensure you have `Poker Hand Classification Model.txt` (it is recommended to rename it to `poker_hand_classifier.py`).

2.  **Set up the dataset:**
    -   Create the folder `C:\Users\YourUser\Documents\Poker Hand Analysis`.
    -   Download the dataset files (`poker-hand-training.csv` and `poker-hand-testing.csv`) and place them in this folder.

3.  **Install dependencies:**
    Create a file named `requirements.txt` in the project directory with the following content:
    ```
    numpy
    pandas
    scikit-learn
    ```
    Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the script, navigate to the project directory in your terminal and execute the Python file.

```bash
python "Poker Hand Classification Model.txt"
