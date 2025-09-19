import numpy as np
from sklearn.ensemble import RandomForestClassifier # Changed from Regressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import ast
from sklearn.metrics import accuracy_score, classification_report # Changed metrics
import os

class PokerHandAnalyzer:
    def __init__(self):
        # Changed to a Classifier for predicting hand categories
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def preprocess_hand(self, cards, position=0, stack_size=0, pot_size=0): # Default values for unused features
        """Preprocess hand data into numerical features"""
        # This function is now used just for card features, as the new dataset doesn't have position/stack/pot.
        
        # Extract ranks and suits, handling '10' as a two-character rank
        ranks = []
        suits = []
        for card in cards:
            if card.startswith('10'):
                ranks.append(card[:2]) # Rank is '10'
                suits.append(card[2:])  # Suit is the rest
            else:
                ranks.append(card[:1]) # Rank is the first char
                suits.append(card[1:])  # Suit is the rest
        
        # Encode ranks (A=14, K=13, Q=12, J=11, T=10)
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '10': 10}
        
        encoded_ranks = []
        for rank in ranks:
            if rank in rank_map:
                encoded_ranks.append(rank_map[rank])
            else:
                try:
                    # For ranks '2' through '9'
                    encoded_ranks.append(int(rank))
                except ValueError:
                    encoded_ranks.append(0) # Should not happen with clean data

        # Encode suits (H=1, S=2, D=3, C=4 from the dataset)
        suit_map = {'H': 1, 'S': 2, 'D': 3, 'C': 4}
        encoded_suits = [suit_map.get(suit, 0) for suit in suits]

        # Create features from the 5 cards
        features = [
            np.mean(encoded_ranks),           # Average rank
            np.std(encoded_ranks),            # Rank standard deviation
            len(set(encoded_suits)),          # Number of unique suits
            max(encoded_ranks) - min(encoded_ranks),  # Rank spread
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        
    def predict_hand_type(self, cards):
        """Predict poker hand type"""
        features = self.preprocess_hand(cards)
        return self.model.predict(features)[0]

def load_and_preprocess_data(file_path, analyzer):
    """Loads data from the new CSV format and preprocesses it."""
    try:
        # The new CSV uses tabs as separators and has no header
        data_df = pd.read_csv(file_path, sep=',', header=0)
        # Rename columns to be more user-friendly based on the provided header
        data_df.columns = [
            'Suit1', 'Rank1', 'Suit2', 'Rank2', 'Suit3', 'Rank3', 
            'Suit4', 'Rank4', 'Suit5', 'Rank5', 'PokerHand'
        ]
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Please ensure the dataset file is in the correct directory.")
        return None, None

    X_list = []
    y_list = []
    
    # Rank and Suit maps for creating card strings
    rank_map_inv = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
    suit_map_inv = {1: 'H', 2: 'S', 3: 'D', 4: 'C'} # Adjusted to common suit mapping

    for index, row in data_df.iterrows():
        cards_list = []
        for i in range(1, 6):
            rank = rank_map_inv.get(row[f'Rank{i}'], str(row[f'Rank{i}']))
            suit = suit_map_inv.get(row[f'Suit{i}'], 'X') # 'X' for unknown suit
            cards_list.append(f"{rank}{suit}")

        features = analyzer.preprocess_hand(cards_list)
        X_list.append(features.flatten())
        y_list.append(row['PokerHand'])
        
    return np.array(X_list), np.array(y_list)

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PokerHandAnalyzer()
    
    # --- Define file paths ---
    home_dir = os.path.expanduser('~')
    data_folder_path = os.path.join(home_dir, 'Documents', 'Poker Hand Analysis')
    
    # Make sure your files are named this way
    train_data_path = os.path.join(data_folder_path, 'poker-hand-training.csv')
    test_data_path = os.path.join(data_folder_path, 'poker-hand-testing.csv')

    # --- Load and Preprocess Training and Testing Data ---
    print("Loading and preprocessing data...")
    X_train, y_train = load_and_preprocess_data(train_data_path, analyzer)
    X_test, y_test = load_and_preprocess_data(test_data_path, analyzer)

    # Exit if data loading failed
    if X_train is None or X_test is None:
        exit()

    # --- Train the Model ---
    print(f"Training model with {len(X_train)} hands...")
    analyzer.train(X_train, y_train)
    print("Training complete.")

    # --- Evaluate Model Performance ---
    print("\n--- Evaluating Model Performance ---")
    y_pred = analyzer.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    # Define labels for the report
    hand_names = [
        "Nothing", "One pair", "Two pairs", "Three of a kind", 
        "Straight", "Flush", "Full house", "Four of a kind", 
        "Straight flush", "Royal flush"
    ]
    # The unique labels in y_test might not include all 0-9, so we get them dynamically
    target_labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
    target_names = [hand_names[i] for i in target_labels]
    print(classification_report(y_test, y_pred, labels=target_labels, target_names=target_names))

    # --- Making a Prediction on a New Hand ---
    print("\n--- Making a Prediction on a New Hand ---")
    # Example hand: A Full House (Aces over Kings)
    # Each card must be a 2+ character string: Rank + Suit
    example_hand = ['AS', 'AD', 'AC', 'KH', 'KD'] 
    
    predicted_hand_code = analyzer.predict_hand_type(example_hand)
    predicted_hand_name = hand_names[predicted_hand_code]

    print(f"Hand to predict: {example_hand}")
    print(f"Predicted Hand Type: {predicted_hand_name} (Code: {predicted_hand_code})")
