"""
Inference module for sentiment analysis using fine-tuned BERT model.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from pathlib import Path


class SentimentModel:
    """
    Sentiment analysis model for book reviews.
    Predicts sentiment scores from 1-5 based on review text.
    """
    
    def __init__(self, model_path="models/sentiment_model", max_length=512, device=None):
        """
        Initialize the sentiment model.
        
        Args:
            model_path: Path to the saved model directory
            max_length: Maximum sequence length for tokenization
            device: PyTorch device (None for auto-detection)
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize model and tokenizer as None (lazy loading)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def _load_model(self):
        """Load model and tokenizer from disk."""
        if self._is_loaded:
            return
        
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Please run train_model.py first to train and save the model."
            )
        
        print(f"Loading model from {self.model_path}...")
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self._is_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict_sentiment(self, text, return_probs=False):
        """
        Predict sentiment score (1-5) for a given review text.
        
        Args:
            text: Review text string
            return_probs: If True, also return probability distribution
        
        Returns:
            Integer score (1-5) or tuple (score, probabilities) if return_probs=True
        """
        # Load model if not already loaded
        if not self._is_loaded:
            self._load_model()
        
        # Preprocess text
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Input text must be a non-empty string")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class (0-4) and convert to score (1-5)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        score = predicted_class + 1  # Convert from 0-4 to 1-5
        
        # Ensure score is in valid range
        score = max(1, min(5, score))
        
        if return_probs:
            probs = probabilities[0].cpu().numpy()
            return score, probs
        else:
            return score
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict sentiment scores for multiple texts.
        
        Args:
            texts: List of review text strings
            batch_size: Batch size for processing
        
        Returns:
            List of integer scores (1-5)
        """
        # Load model if not already loaded
        if not self._is_loaded:
            self._load_model()
        
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted classes and convert to scores
            predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy()
            batch_scores = (predicted_classes + 1).tolist()  # Convert from 0-4 to 1-5
            
            # Ensure scores are in valid range
            batch_scores = [max(1, min(5, s)) for s in batch_scores]
            scores.extend(batch_scores)
        
        return scores
