import argparse
import logging
from datetime import datetime

from misinformation_detection.data_collection.twitter import (
    TwitterCollector,
    collect_sample_tweets,
)
from misinformation_detection.processing.text_processing import TextProcessor
from misinformation_detection.processing.image_processing import ImageProcessor
from misinformation_detection.analysis.detection import MisinformationDetector
from misinformation_detection.visualization.dashboard import run_dashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=f'misinformation_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Social Media Misinformation Detection System"
    )
    parser.add_argument(
        "--collect-twitter", action="store_true", help="Collect data from Twitter"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Run the Streamlit dashboard"
    )
    parser.add_argument("--sample", action="store_true", help="Run with sample data")

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_args()

    logger.info("Starting Social Media Misinformation Detection System")

    if args.collect_twitter:
        logger.info("Running Twitter data collection")
        collector = TwitterCollector()
        health_keywords = [
            "covid vaccine",
            "health supplements",
            "weight loss diet",
            "alternative medicine",
            "natural remedies",
        ]
        collector.collect_health_tweets(health_keywords)

    if args.dashboard:
        logger.info("Starting Streamlit dashboard")
        run_dashboard()

    if args.sample:
        logger.info("Running with sample data")
        collect_sample_tweets()

    if not any([args.collect_twitter, args.dashboard, args.sample]):
        logger.info("No specific action specified. Use --help for available options.")
        print("Usage instructions:")
        print("  --collect-twitter    Collect data from Twitter")
        print("  --dashboard          Run the Streamlit dashboard")
        print("  --sample             Run with sample data")


if __name__ == "__main__":
    main()
