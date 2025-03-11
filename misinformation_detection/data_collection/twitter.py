import time
import os
import logging
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

import tweepy
from sqlalchemy.orm import Session

from config.config import (
    TWITTER_API_KEY, 
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
    TWITTER_TWEETS_DIR,
    TWITTER_TEXT_ONLY_DIR,
    TWITTER_WITH_IMAGES_DIR,
    TWITTER_IMAGES_DIR,
    TWITTER_VIDEOS_DIR
)
from misinformation_detection.database import get_db
from misinformation_detection.models import Tweet, TwitterMedia

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwitterCollector:
    """Class for collecting data from Twitter API."""

    def __init__(self):
        """Initialize the Twitter API client."""
        try:
            auth = tweepy.OAuth1UserHandler(
                TWITTER_API_KEY,
                TWITTER_API_SECRET,
                TWITTER_ACCESS_TOKEN,
                TWITTER_ACCESS_SECRET
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            self.client = tweepy.Client(
                consumer_key=TWITTER_API_KEY,
                consumer_secret=TWITTER_API_SECRET,
                access_token=TWITTER_ACCESS_TOKEN,
                access_token_secret=TWITTER_ACCESS_SECRET
            )
            logger.info("Twitter API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            raise

    def search_tweets(
        self, 
        query: str, 
        count: int = 100, 
        result_type: str = "mixed"
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets matching the given query.
        
        Args:
            query: Search query string
            count: Maximum number of tweets to return
            result_type: Type of results (recent, popular, or mixed)
            
        Returns:
            List of tweet data dictionaries
        """
        try:
            tweets = self.api.search_tweets(
                q=query,
                count=count,
                result_type=result_type,
                tweet_mode="extended"
            )
            logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            return tweets
        except tweepy.TweepyException as e:
            logger.error(f"Error searching tweets: {e}")
            # Implement exponential backoff
            if "rate limit" in str(e).lower():
                wait_time = 60  # Start with 1 minute
                for attempt in range(3):  # Try 3 times
                    logger.info(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    try:
                        tweets = self.api.search_tweets(
                            q=query,
                            count=count,
                            result_type=result_type,
                            tweet_mode="extended"
                        )
                        logger.info(f"Retrieved {len(tweets)} tweets after waiting")
                        return tweets
                    except tweepy.TweepyException:
                        wait_time *= 2  # Double the wait time
            return []

    def download_media(self, media_url: str, tweet_id: str, media_type: str) -> Optional[str]:
        """
        Download media from a URL and save it locally.
        
        Args:
            media_url: URL of the media to download
            tweet_id: ID of the tweet containing the media
            media_type: Type of media (photo, video, etc.)
            
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            # Create a filename from the URL
            filename = os.path.basename(media_url)
            # Add tweet_id to ensure uniqueness
            
            # Determine destination directory based on media type
            if media_type == "photo":
                dest_dir = TWITTER_IMAGES_DIR
            elif media_type == "video":
                dest_dir = TWITTER_VIDEOS_DIR
            else:
                dest_dir = TWITTER_IMAGES_DIR  # Default to images directory
                
            filepath = os.path.join(dest_dir, f"{tweet_id}_{filename}")
            
            # Download the media
            response = requests.get(media_url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                logger.info(f"Downloaded media to {filepath}")
                return filepath
            else:
                logger.error(f"Failed to download media: HTTP {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None

    def save_tweet_to_db(self, tweet, db: Session) -> Tweet:
        """
        Save a tweet and its media to the database.
        
        Args:
            tweet: Tweepy Status object
            db: Database session
            
        Returns:
            Saved Tweet model instance
        """
        try:
            # Extract tweet data
            tweet_data = {
                "id": tweet.id_str,
                "content": tweet.full_text if hasattr(tweet, "full_text") else tweet.text,
                "author_id": tweet.user.id_str,
                "author_username": tweet.user.screen_name,
                "created_at": tweet.created_at,
                "like_count": tweet.favorite_count if hasattr(tweet, "favorite_count") else 0,
                "retweet_count": tweet.retweet_count if hasattr(tweet, "retweet_count") else 0,
                "reply_count": getattr(tweet, "reply_count", 0),
                "quote_count": getattr(tweet, "quote_count", 0),
            }
            
            # Check if the tweet has media
            has_media = hasattr(tweet, "extended_entities") and "media" in tweet.extended_entities
            tweet_data["has_media"] = has_media
            
            # Create Tweet model instance
            db_tweet = Tweet(**tweet_data)
            db.add(db_tweet)
            db.commit()
            
            # Save tweet to appropriate file based on media presence
            self.save_tweet_to_file(tweet, has_media)
            
            # Process media if present
            if has_media:
                for media in tweet.extended_entities["media"]:
                    media_url = media["media_url_https"]
                    media_type = media["type"]
                    
                    # Download media file
                    local_path = self.download_media(media_url, tweet.id_str, media_type)
                    
                    # Create TwitterMedia model instance
                    db_media = TwitterMedia(
                        id=media["id_str"],
                        tweet_id=tweet.id_str,
                        media_type=media_type,
                        url=media_url,
                        local_path=local_path
                    )
                    db.add(db_media)
            
            db.commit()
            logger.info(f"Saved tweet {tweet.id_str} to database")
            return db_tweet
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving tweet to database: {e}")
            raise
            
    def save_tweet_to_file(self, tweet, has_media: bool) -> None:
        """
        Save tweet to appropriate file based on whether it has media.
        
        Args:
            tweet: Tweepy Status object
            has_media: Whether the tweet has media
        """
        try:
            # Create a simplified JSON representation of the tweet
            tweet_json = {
                "id": tweet.id_str,
                "text": tweet.full_text if hasattr(tweet, "full_text") else tweet.text,
                "author": tweet.user.screen_name,
                "created_at": tweet.created_at.isoformat(),
                "like_count": tweet.favorite_count if hasattr(tweet, "favorite_count") else 0,
                "retweet_count": tweet.retweet_count if hasattr(tweet, "retweet_count") else 0,
            }
            
            # Save to the general tweets directory
            general_path = os.path.join(TWITTER_TWEETS_DIR, f"{tweet.id_str}.json")
            with open(general_path, 'w') as f:
                json.dump(tweet_json, f, indent=2)
            
            # Save to the appropriate category directory
            if has_media:
                category_path = os.path.join(TWITTER_WITH_IMAGES_DIR, f"{tweet.id_str}.json")
            else:
                category_path = os.path.join(TWITTER_TEXT_ONLY_DIR, f"{tweet.id_str}.json")
                
            with open(category_path, 'w') as f:
                json.dump(tweet_json, f, indent=2)
                
            logger.info(f"Saved tweet {tweet.id_str} to files")
        except Exception as e:
            logger.error(f"Error saving tweet to file: {e}")

    def collect_health_tweets(self, keywords: List[str], count_per_keyword: int = 20):
        """
        Collect tweets related to health topics.
        
        Args:
            keywords: List of health-related keywords to search for
            count_per_keyword: Number of tweets to collect per keyword
        """
        for keyword in keywords:
            try:
                logger.info(f"Collecting tweets for keyword: {keyword}")
                tweets = self.search_tweets(keyword, count=count_per_keyword)
                
                # Save tweets to database
                db = next(get_db())
                for tweet in tweets:
                    # Check if tweet already exists
                    existing = db.query(Tweet).filter(Tweet.id == tweet.id_str).first()
                    if not existing:
                        self.save_tweet_to_db(tweet, db)
                    else:
                        logger.info(f"Tweet {tweet.id_str} already exists in database")
                
                logger.info(f"Finished collecting tweets for keyword: {keyword}")
            except Exception as e:
                logger.error(f"Error collecting tweets for keyword {keyword}: {e}")


# Example usage function
def collect_sample_tweets():
    """Collect a sample of health-related tweets."""
    health_keywords = [
        "covid vaccine",
        "health supplements",
        "weight loss diet",
        "alternative medicine",
        "natural remedies"
    ]
    
    collector = TwitterCollector()
    collector.collect_health_tweets(health_keywords, count_per_keyword=10)


if __name__ == "__main__":
    collect_sample_tweets()
