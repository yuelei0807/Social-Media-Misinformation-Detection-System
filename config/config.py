import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/misinformation.db")

# Twitter API configuration
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")

# Data directories
DATA_DIR = BASE_DIR / "data"
TWITTER_TWEETS_DIR = DATA_DIR / "twitter_tweets"
TWITTER_TEXT_ONLY_DIR = DATA_DIR / "twitter_text_only"
TWITTER_WITH_IMAGES_DIR = DATA_DIR / "twitter_with_images"
TWITTER_IMAGES_DIR = DATA_DIR / "twitter_images"
TWITTER_VIDEOS_DIR = DATA_DIR / "twitter_videos"
TWITTER_FRAMES_DIR = DATA_DIR / "twitter_frames"
TIKTOK_VIDEOS_DIR = DATA_DIR / "tiktok_videos"
TIKTOK_FRAMES_DIR = DATA_DIR / "tiktok_frames"

# Ensure directories exist
for directory in [
    TWITTER_TWEETS_DIR,
    TWITTER_TEXT_ONLY_DIR,
    TWITTER_WITH_IMAGES_DIR,
    TWITTER_IMAGES_DIR,
    TWITTER_VIDEOS_DIR,
    TWITTER_FRAMES_DIR,
    TIKTOK_VIDEOS_DIR,
    TIKTOK_FRAMES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
