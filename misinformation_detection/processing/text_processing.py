import re
import string
import logging
from typing import List, Dict, Any

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


class TextProcessor:
    """Class for processing and analyzing text content."""

    def __init__(self):
        """Initialize the text processor."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        logger.info("TextProcessor initialized")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, hashtags, and special characters.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (#topic) - keep the topic text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT markers
        text = re.sub(r'^RT\s+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens

    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text for analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        
        if not tokens:
            return {
                "word_count": 0,
                "avg_word_length": 0,
                "unique_words_ratio": 0,
                "tokens": [],
                "cleaned_text": ""
            }
        
        # Calculate basic statistics
        word_count = len(tokens)
        avg_word_length = sum(len(t) for t in tokens) / word_count if word_count > 0 else 0
        unique_words_ratio = len(set(tokens)) / word_count if word_count > 0 else 0
        
        features = {
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "unique_words_ratio": unique_words_ratio,
            "tokens": tokens,
            "cleaned_text": cleaned_text
        }
        
        return features

    def identify_health_topics(self, text: str) -> List[str]:
        """
        Identify health-related topics in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified health topics
        """
        # This is a simple keyword-based approach
        # In a real implementation, you would use a more sophisticated method
        
        health_topics = {
            "covid": ["covid", "coronavirus", "pandemic", "vaccine", "vaccination", "pfizer", "moderna", "astrazeneca"],
            "diet": ["diet", "nutrition", "weight loss", "calories", "keto", "carbs", "fat"],
            "supplements": ["supplement", "vitamin", "mineral", "herbal", "natural remedy"],
            "alternative medicine": ["alternative", "holistic", "homeopathy", "naturopathy", "acupuncture"],
            "mental health": ["mental health", "depression", "anxiety", "stress", "therapy"]
        }
        
        cleaned_text = self.clean_text(text).lower()
        identified_topics = []
        
        for topic, keywords in health_topics.items():
            if any(keyword in cleaned_text for keyword in keywords):
                identified_topics.append(topic)
        
        return identified_topics


# Example usage function
def process_sample_text():
    """Process a sample text to demonstrate text processing."""
    sample_text = """
    RT @health_expert: New study shows that Vitamin D supplements may help prevent #COVID19 infections. 
    Read more at https://example.com/study #health #vitamins
    """
    
    processor = TextProcessor()
    
    # Clean and tokenize
    cleaned_text = processor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned_text}")
    
    tokens = processor.tokenize(cleaned_text)
    print(f"Tokens: {tokens}")
    
    # Extract features
    features = processor.extract_features(sample_text)
    print(f"Features: {features}")
    
    # Identify health topics
    topics = processor.identify_health_topics(sample_text)
    print(f"Health topics: {topics}")


if __name__ == "__main__":
    process_sample_text()
