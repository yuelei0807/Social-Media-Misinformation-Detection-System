import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

from misinformation_detection.processing.text_processing import TextProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MisinformationDetector:
    """Class for detecting potential misinformation in social media content."""

    def __init__(self):
        """Initialize the misinformation detector."""
        self.text_processor = TextProcessor()

        # Load pre-trained model for fake news detection
        try:
            # Note: In a real implementation, you'd use a health misinformation-specific model
            # The model name below is just an example and might not exist
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Fallback to sentiment model as example
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                "text-classification", model=self.model, tokenizer=self.tokenizer
            )
            logger.info(f"Loaded classification model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            logger.warning("Using fallback rule-based detection")
            self.classifier = None

        # Health misinformation keywords and patterns
        self.suspicious_patterns = {
            "miracle_cure": [
                "miracle cure",
                "miracle treatment",
                "miraculous",
                "100% effective",
                "guaranteed",
                "secret cure",
                "doctors hate this",
                "big pharma doesn't want you to know",
            ],
            "conspiracy": [
                "conspiracy",
                "cover up",
                "they don't want you to know",
                "hidden truth",
                "suppressed",
                "what they're hiding",
            ],
            "exaggeration": [
                "revolutionary",
                "breakthrough",
                "game-changer",
                "life-changing",
                "never seen before",
                "shocking results",
            ],
            "quick_results": [
                "instant results",
                "overnight",
                "immediate",
                "rapid",
                "quick",
                "fast results",
                "lose weight fast",
            ],
            "unverified_science": [
                "studies show",
                "research proves",
                "scientifically proven",
                "clinically proven",
                "doctors recommend",
                "experts agree",
            ],
        }

        logger.info("MisinformationDetector initialized")

    def analyze_text_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text content for potential misinformation.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary of analysis results
        """
        try:
            # Clean and process text
            features = self.text_processor.extract_features(text)
            cleaned_text = features["cleaned_text"]

            # Use model-based detection if available
            model_score = self._get_model_score(cleaned_text)

            # Use rule-based detection as a fallback or supplement
            rule_based_results = self._check_suspicious_patterns(cleaned_text)
            rule_score = rule_based_results["score"]

            # Combine scores (weighted average)
            if model_score is not None:
                # If model score is available, give it more weight
                combined_score = (model_score * 0.7) + (rule_score * 0.3)
            else:
                # Otherwise, use only rule-based score
                combined_score = rule_score

            # Health topics identification
            health_topics = self.text_processor.identify_health_topics(text)

            analysis = {
                "model_score": model_score,
                "rule_based_score": rule_score,
                "combined_score": combined_score,
                "suspicious_patterns": rule_based_results["patterns"],
                "is_potential_misinformation": combined_score > 0.6,  # Threshold
                "confidence": min(max(combined_score, 0.0), 1.0),  # Ensure 0-1 range
                "health_topics": health_topics,
                "explanation": self._generate_explanation(
                    combined_score, rule_based_results["patterns"], health_topics
                ),
            }

            logger.info(
                f"Text analysis completed, misinformation score: {combined_score:.2f}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing text content: {e}")
            return {
                "error": str(e),
                "combined_score": 0.0,
                "is_potential_misinformation": False,
                "explanation": "Error occurred during analysis",
            }

    def analyze_image_results(self, image_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze image processing results for potential misinformation indicators.

        Args:
            image_analysis: Image analysis results

        Returns:
            Dictionary of misinformation analysis for the image
        """
        try:
            # Extract relevant features from image analysis
            text_present = image_analysis.get("text_present", False)
            text_content = image_analysis.get("text_content", "")
            manipulation_score = image_analysis.get("manipulation_indicators", {}).get(
                "score", 0.0
            )

            # If text is present, analyze it for misinformation
            text_score = 0.0
            suspicious_patterns = []
            if text_present and text_content:
                text_analysis = self.analyze_text_content(text_content)
                text_score = text_analysis.get("combined_score", 0.0)
                suspicious_patterns = text_analysis.get("suspicious_patterns", [])

            # Combine text and manipulation scores
            # Weigh more heavily toward text content if it exists
            if text_present:
                combined_score = (text_score * 0.7) + (manipulation_score * 0.3)
            else:
                combined_score = manipulation_score

            # Generate analysis results
            analysis = {
                "text_score": text_score if text_present else None,
                "manipulation_score": manipulation_score,
                "combined_score": combined_score,
                "suspicious_patterns": suspicious_patterns,
                "is_potential_misinformation": combined_score > 0.6,  # Threshold
                "confidence": min(max(combined_score, 0.0), 1.0),  # Ensure 0-1 range
                "explanation": self._generate_image_explanation(
                    combined_score,
                    text_present,
                    suspicious_patterns,
                    manipulation_score,
                ),
            }

            logger.info(
                f"Image analysis completed, misinformation score: {combined_score:.2f}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing image results: {e}")
            return {
                "error": str(e),
                "combined_score": 0.0,
                "is_potential_misinformation": False,
                "explanation": "Error occurred during analysis",
            }

    def combine_content_analyses(
        self,
        text_analysis: Optional[Dict[str, Any]] = None,
        image_analyses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Combine analyses from text and images for an overall misinformation score.

        Args:
            text_analysis: Text content analysis results
            image_analyses: List of image analysis results

        Returns:
            Combined analysis results
        """
        try:
            # Initialize scores
            text_score = 0.0
            image_score = 0.0

            # Process text analysis if available
            if text_analysis:
                text_score = text_analysis.get("combined_score", 0.0)

            # Process image analyses if available
            if image_analyses and len(image_analyses) > 0:
                # Average the image scores
                image_scores = [
                    img.get("combined_score", 0.0) for img in image_analyses
                ]
                image_score = sum(image_scores) / len(image_scores)

                # Get all suspicious patterns from images
                image_patterns = []
                for img in image_analyses:
                    patterns = img.get("suspicious_patterns", [])
                    image_patterns.extend(patterns)
            else:
                image_patterns = []

            # Combine text and image scores
            # If both are available, weigh them evenly
            # Otherwise, use whichever is available
            if text_analysis and image_analyses:
                combined_score = (text_score + image_score) / 2
                all_patterns = (
                    text_analysis.get("suspicious_patterns", []) + image_patterns
                )
            elif text_analysis:
                combined_score = text_score
                all_patterns = text_analysis.get("suspicious_patterns", [])
            elif image_analyses:
                combined_score = image_score
                all_patterns = image_patterns
            else:
                combined_score = 0.0
                all_patterns = []

            # Remove duplicate patterns
            unique_patterns = list(set(all_patterns))

            # Generate combined analysis
            analysis = {
                "text_score": text_score if text_analysis else None,
                "image_score": image_score if image_analyses else None,
                "combined_score": combined_score,
                "suspicious_patterns": unique_patterns,
                "is_potential_misinformation": combined_score > 0.6,  # Threshold
                "confidence": min(max(combined_score, 0.0), 1.0),  # Ensure 0-1 range
                "explanation": self._generate_combined_explanation(
                    combined_score,
                    text_analysis is not None,
                    image_analyses is not None and len(image_analyses) > 0,
                    unique_patterns,
                ),
            }

            logger.info(
                f"Combined analysis completed, misinformation score: {combined_score:.2f}"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error combining content analyses: {e}")
            return {
                "error": str(e),
                "combined_score": 0.0,
                "is_potential_misinformation": False,
                "explanation": "Error occurred during analysis",
            }

    def _get_model_score(self, text: str) -> Optional[float]:
        """
        Get misinformation score from the pre-trained model.

        Args:
            text: Cleaned text content

        Returns:
            Misinformation score between 0 and 1, or None if model is unavailable
        """
        if not self.classifier or not text:
            return None

        try:
            # Truncate text if too long for the model
            max_length = self.tokenizer.model_max_length
            tokens = self.tokenizer.encode(text, truncation=False)
            if len(tokens) > max_length:
                logger.info(
                    f"Truncating text from {len(tokens)} tokens to {max_length}"
                )
                tokens = tokens[:max_length]
                text = self.tokenizer.decode(tokens)

            # Get model prediction
            result = self.classifier(text)[0]

            # Extract score based on label
            label = result["label"]
            score = result["score"]

            # Convert to misinformation score (0-1)
            # For this example model, we'll assume "NEGATIVE" is more likely to be misinformation
            # In a real implementation with a misinformation-specific model, this would be different
            if label == "NEGATIVE":
                return score
            else:
                return 1.0 - score
        except Exception as e:
            logger.error(f"Error getting model score: {e}")
            return None

    def _check_suspicious_patterns(self, text: str) -> Dict[str, Any]:
        """
        Check text for suspicious patterns indicative of misinformation.

        Args:
            text: Cleaned text content

        Returns:
            Dictionary with pattern matches and score
        """
        if not text:
            return {"patterns": [], "score": 0.0}

        text_lower = text.lower()
        matched_patterns = []

        # Check each pattern category
        for category, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    matched_patterns.append(f"{category}: {pattern}")

        # Calculate score based on number of matches
        # More matches = higher score
        score = min(len(matched_patterns) / 5.0, 1.0)  # Cap at 1.0

        return {"patterns": matched_patterns, "score": score}

    def _generate_explanation(
        self, score: float, patterns: List[str], health_topics: List[str]
    ) -> str:
        """
        Generate a human-readable explanation for the misinformation analysis.

        Args:
            score: Combined misinformation score
            patterns: List of suspicious patterns found
            health_topics: Health topics identified in the content

        Returns:
            Explanation text
        """
        if score > 0.8:
            confidence = "high"
        elif score > 0.6:
            confidence = "moderate"
        elif score > 0.4:
            confidence = "low"
        else:
            confidence = "very low"

        explanation = f"This content has a {confidence} likelihood of containing health misinformation "
        explanation += f"(score: {score:.2f}).\n\n"

        if health_topics:
            explanation += f"Health topics detected: {', '.join(health_topics)}.\n\n"

        if patterns:
            explanation += "Potential misinformation indicators:\n"
            for pattern in patterns:
                explanation += f"- {pattern}\n"
        else:
            explanation += "No specific misinformation indicators were detected."

        if score <= 0.4:
            explanation += "\n\nThis content appears to be relatively reliable based on our analysis."
        elif score <= 0.6:
            explanation += "\n\nThis content contains some potential misinformation indicators but requires further verification."
        else:
            explanation += "\n\nThis content contains multiple indicators associated with health misinformation."

        return explanation

    def _generate_image_explanation(
        self,
        score: float,
        text_present: bool,
        patterns: List[str],
        manipulation_score: float,
    ) -> str:
        """
        Generate a human-readable explanation for the image misinformation analysis.

        Args:
            score: Combined misinformation score
            text_present: Whether text was found in the image
            patterns: List of suspicious patterns found in the text
            manipulation_score: Image manipulation score

        Returns:
            Explanation text
        """
        if score > 0.8:
            confidence = "high"
        elif score > 0.6:
            confidence = "moderate"
        elif score > 0.4:
            confidence = "low"
        else:
            confidence = "very low"

        explanation = f"This image has a {confidence} likelihood of containing health misinformation "
        explanation += f"(score: {score:.2f}).\n\n"

        if manipulation_score > 0.7:
            explanation += (
                "The image shows strong indicators of manipulation or editing.\n"
            )
        elif manipulation_score > 0.4:
            explanation += (
                "The image shows some indicators of possible manipulation or editing.\n"
            )
        else:
            explanation += "The image does not show clear signs of manipulation.\n"

        if text_present:
            explanation += "\nText was detected in the image. "
            if patterns:
                explanation += (
                    "The following suspicious patterns were found in the text:\n"
                )
                for pattern in patterns:
                    explanation += f"- {pattern}\n"
            else:
                explanation += "No suspicious patterns were found in the text."

        return explanation

    def _generate_combined_explanation(
        self, score: float, has_text: bool, has_images: bool, patterns: List[str]
    ) -> str:
        """
        Generate a human-readable explanation for the combined analysis.

        Args:
            score: Combined misinformation score
            has_text: Whether text content was analyzed
            has_images: Whether image content was analyzed
            patterns: List of suspicious patterns found

        Returns:
            Explanation text
        """
        if score > 0.8:
            confidence = "high"
        elif score > 0.6:
            confidence = "moderate"
        elif score > 0.4:
            confidence = "low"
        else:
            confidence = "very low"

        explanation = f"This content has a {confidence} likelihood of containing health misinformation "
        explanation += f"(score: {score:.2f}).\n\n"

        content_types = []
        if has_text:
            content_types.append("text")
        if has_images:
            content_types.append("images")

        explanation += f"Analysis based on: {' and '.join(content_types)}.\n\n"

        if patterns:
            explanation += "Potential misinformation indicators:\n"
            for pattern in patterns:
                explanation += f"- {pattern}\n"
        else:
            explanation += "No specific misinformation indicators were detected."

        if score <= 0.4:
            explanation += "\n\nThis content appears to be relatively reliable based on our analysis."
        elif score <= 0.6:
            explanation += "\n\nThis content contains some potential misinformation indicators but requires further verification."
        else:
            explanation += "\n\nThis content contains multiple indicators associated with health misinformation."

        explanation += "\n\nNote: This is an automated analysis and should not be considered definitive. Always verify health information with trusted sources."

        return explanation


# Example usage function
def detect_sample_misinformation():
    """Detect misinformation in sample content."""
    sample_text = """
    BREAKING: Secret cure for COVID they don't want you to know about!
    Scientists have discovered that drinking alkaline water mixed with lemon juice
    ELIMINATES the coronavirus in just 24 hours! Big pharma doesn't want this getting out
    because it's 100% effective and they can't profit from it. Share this MIRACLE CURE before
    they take it down!!!
    """

    # Sample image analysis results (simulated)
    sample_image_analysis = {
        "text_present": True,
        "text_content": "Doctors SHOCKED by this natural COVID cure! 100% effective!",
        "manipulation_indicators": {"score": 0.7, "suspicious": True},
    }

    detector = MisinformationDetector()

    # Analyze text
    text_analysis = detector.analyze_text_content(sample_text)
    print(f"Text analysis results:")
    print(f"Score: {text_analysis['combined_score']:.2f}")
    print(f"Is misinformation: {text_analysis['is_potential_misinformation']}")
    print(f"Explanation: {text_analysis['explanation']}")

    # Analyze image
    image_analysis = detector.analyze_image_results(sample_image_analysis)
    print(f"\nImage analysis results:")
    print(f"Score: {image_analysis['combined_score']:.2f}")
    print(f"Is misinformation: {image_analysis['is_potential_misinformation']}")
    print(f"Explanation: {image_analysis['explanation']}")

    # Combine analyses
    combined_analysis = detector.combine_content_analyses(
        text_analysis=text_analysis, image_analyses=[image_analysis]
    )
    print(f"\nCombined analysis results:")
    print(f"Score: {combined_analysis['combined_score']:.2f}")
    print(f"Is misinformation: {combined_analysis['is_potential_misinformation']}")
    print(f"Explanation: {combined_analysis['explanation']}")


if __name__ == "__main__":
    detect_sample_misinformation()
