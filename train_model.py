"""
Training script for fine-tuning BERT model on book review sentiment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import os
from pathlib import Path


class ReviewDataset(Dataset):
    """PyTorch Dataset for book reviews."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss to handle class imbalance."""
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move class weights to the same device as the model
        device = next(model.parameters()).device
        class_weights = self.class_weights.to(device)
        
        # Use weighted cross entropy loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def load_and_preprocess_data(ratings_file="data/ratings.csv", sample_size=None):
    """
    Load and preprocess ratings data.
    
    Args:
        ratings_file: Path to ratings CSV file
        sample_size: Optional limit on number of samples (for faster training)
    
    Returns:
        DataFrame with cleaned text and integer scores
    """
    print("Loading data...")
    df = pd.read_csv(ratings_file)
    
    # Extract relevant columns
    df = df[['review/text', 'review/score']].copy()
    
    # Remove rows with missing values
    df = df.dropna(subset=['review/text', 'review/score'])
    
    # Convert scores to integers (1-5)
    df['review/score'] = df['review/score'].astype(float).astype(int)
    
    # Filter valid scores (1-5)
    df = df[df['review/score'].between(1, 5)]
    
    # Clean text: remove empty strings
    df['review/text'] = df['review/text'].astype(str)
    df = df[df['review/text'].str.strip() != '']
    
    # Sample if specified
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} reviews for training")
    
    print(f"Loaded {len(df)} reviews")
    print(f"Score distribution:\n{df['review/score'].value_counts().sort_index()}")
    
    return df


def prepare_datasets(df, tokenizer, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare train, validation, and test datasets.
    
    Args:
        df: DataFrame with 'review/text' and 'review/score' columns
        tokenizer: BERT tokenizer
        test_size: Proportion for test set
        val_size: Proportion of training set for validation
        random_state: Random seed
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, y_test, train_labels)
    """
    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['review/score']
    )
    
    # Split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['review/score']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get training labels for class weight calculation
    train_labels = (train_df['review/score'].values - 1).tolist()  # Convert to 0-4
    
    # Create datasets
    train_dataset = ReviewDataset(
        train_df['review/text'].values,
        train_labels,
        tokenizer
    )
    
    val_dataset = ReviewDataset(
        val_df['review/text'].values,
        (val_df['review/score'].values - 1).tolist(),
        tokenizer
    )
    
    test_dataset = ReviewDataset(
        test_df['review/text'].values,
        (test_df['review/score'].values - 1).tolist(),
        tokenizer
    )
    
    return train_dataset, val_dataset, test_dataset, test_df['review/score'].values, train_labels


def calculate_class_weights(labels, device=None, method='sqrt'):
    """
    Calculate class weights to handle imbalanced data.
    
    Args:
        labels: List of training labels (0-4)
        device: PyTorch device
        method: 'balanced' (full inverse frequency), 'sqrt' (square root), or 'log' (logarithmic)
    
    Returns:
        Tensor of class weights
    """
    # Convert to numpy array
    labels_array = np.array(labels)
    
    # Count occurrences of each class
    unique, counts = np.unique(labels_array, return_counts=True)
    total = len(labels_array)
    
    # Calculate weights based on method
    if method == 'balanced':
        # Full inverse frequency weighting (can be too extreme)
        class_weights = compute_class_weight('balanced', classes=unique, y=labels_array)
    elif method == 'sqrt':
        # Square root of inverse frequency (less extreme)
        max_count = counts.max()
        class_weights = np.sqrt(max_count / counts)
    elif method == 'log':
        # Logarithmic scaling (even less extreme)
        max_count = counts.max()
        class_weights = 1 + np.log(max_count / counts)
    else:
        class_weights = np.ones(len(unique))
    
    # Create a weight tensor for all 5 classes
    weight_dict = dict(zip(unique, class_weights))
    weights = torch.tensor([weight_dict.get(i, 1.0) for i in range(5)], dtype=torch.float32)
    
    if device:
        weights = weights.to(device)
    
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class weights ({method}): {weights.tolist()}")
    return weights


def train_model(train_dataset, val_dataset, train_labels, model_name="distilbert-base-uncased", 
                output_dir="models/sentiment_model", num_epochs=3, batch_size=16, 
                learning_rate=2e-5, device=None, class_weight_method='sqrt'):
    """
    Fine-tune BERT model on sentiment classification.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        train_labels: Training labels for class weight calculation
        model_name: Pre-trained model name
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: PyTorch device
    
    Returns:
        Trained model and trainer
    """
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5  # 5 classes for scores 1-5
    )
    
    if device:
        model = model.to(device)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels, device=device, method=class_weight_method)
    
    # Calculate warmup steps based on dataset size (10% of training steps, min 10, max 500)
    num_training_steps = len(train_dataset) // batch_size * num_epochs
    warmup_steps = max(10, min(500, int(num_training_steps * 0.1)))
    print(f"Using {warmup_steps} warmup steps (out of {num_training_steps} total steps)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        warmup_steps=warmup_steps,
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    # Create weighted trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, trainer, tokenizer


def evaluate_model(trainer, test_dataset, y_test_true):
    """
    Evaluate model on test set.
    
    Args:
        trainer: Trained Trainer object
        test_dataset: Test dataset
        y_test_true: True labels (1-5 scale)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1) + 1  # Convert back to 1-5
    
    accuracy = accuracy_score(y_test_true, y_pred)
    cm = confusion_matrix(y_test_true, y_pred)
    report = classification_report(y_test_true, y_pred, target_names=['1', '2', '3', '4', '5'])
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def main():
    """Main training pipeline."""
    # Configuration
    RATINGS_FILE = "data/ratings.csv"
    MODEL_NAME = "distilbert-base-uncased"
    OUTPUT_DIR = "models/sentiment_model"
    SAMPLE_SIZE = 1000  # Use None for full dataset, or set a number for faster training
    # IMPORTANT: Use at least 5000-10000 samples for good results. 500 is too small!
    NUM_EPOCHS = 5  # Increased from 3 for better learning
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    CLASS_WEIGHT_METHOD = 'sqrt'  # 'sqrt' (recommended), 'log', or 'balanced' (more extreme)
    
    # Check for available devices (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CPU)")
    
    # Load and preprocess data
    df = load_and_preprocess_data(RATINGS_FILE, sample_size=SAMPLE_SIZE)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, y_test, train_labels = prepare_datasets(df, tokenizer)
    
    # Train model
    model, trainer, tokenizer = train_model(
        train_dataset,
        val_dataset,
        train_labels,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device,
        class_weight_method=CLASS_WEIGHT_METHOD
    )
    
    # Evaluate
    metrics = evaluate_model(trainer, test_dataset, y_test)
    
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print("\nIMPORTANT NOTES:")
    print("  - If probabilities are all similar (~20% each), the model hasn't learned well")
    print("  - This usually means you need MORE training data (at least 5000-10000 samples)")
    print("  - Check the confusion matrix above to see prediction patterns")
    print("  - If accuracy is low, try increasing SAMPLE_SIZE and NUM_EPOCHS")


if __name__ == "__main__":
    main()
