import os
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from config.config import TIKTOK_VIDEOS_DIR
from misinformation_detection.database import get_db
from misinformation_detection.models import TikTokVideo

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TikTokManualCollector:
    """Class for manually collecting TikTok videos."""

    def __init__(self, collector_name: str):
        """
        Initialize the TikTok manual collector.

        Args:
            collector_name: Name of the team member collecting data
        """
        self.collector_name = collector_name
        logger.info(f"TikTok manual collector initialized for {collector_name}")

    def save_video(
        self,
        video_path: str,
        video_id: str,
        author_id: str,
        author_username: str,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Save a downloaded TikTok video to the project's storage.

        Args:
            video_path: Path to the downloaded video file
            video_id: TikTok video ID
            author_id: TikTok author ID
            author_username: TikTok author username
            description: Video description

        Returns:
            Path to the saved video file if successful, None otherwise
        """
        try:
            # Create a destination path
            extension = os.path.splitext(video_path)[1]
            dest_filename = f"{video_id}{extension}"
            dest_path = os.path.join(TIKTOK_VIDEOS_DIR, dest_filename)

            # Copy the video file
            shutil.copy2(video_path, dest_path)
            logger.info(f"Saved TikTok video to {dest_path}")

            # Save to database
            db = next(get_db())
            try:
                # Check if video already exists
                existing = (
                    db.query(TikTokVideo).filter(TikTokVideo.id == video_id).first()
                )
                if not existing:
                    # Create TikTokVideo model instance
                    db_video = TikTokVideo(
                        id=video_id,
                        author_id=author_id,
                        author_username=author_username,
                        description=description,
                        local_path=dest_path,
                        collected_by=self.collector_name,
                    )
                    db.add(db_video)
                    db.commit()
                    logger.info(f"Saved TikTok video {video_id} to database")
                else:
                    logger.info(f"TikTok video {video_id} already exists in database")
            except Exception as e:
                db.rollback()
                logger.error(f"Error saving TikTok video to database: {e}")
                raise

            return dest_path
        except Exception as e:
            logger.error(f"Error saving TikTok video: {e}")
            return None

    def record_manual_observation(self, video_id: str, data: Dict[str, Any]) -> bool:
        """
        Record manual observation data for a TikTok video.

        Args:
            video_id: TikTok video ID
            data: Observation data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # This is a placeholder for storing manual observations
            # In a real implementation, this would save to a separate database table

            # For now, just log the observation
            logger.info(
                f"Recorded manual observation for TikTok video {video_id}: {data}"
            )
            return True
        except Exception as e:
            logger.error(f"Error recording manual observation: {e}")
            return False


# Example usage function
def sample_manual_collection():
    """Example of how to manually collect TikTok data."""
    collector = TikTokManualCollector(collector_name="team_member_1")

    # This would be a real path to a downloaded video in practice
    sample_video_path = "/path/to/downloaded/tiktok_video.mp4"

    # These would be real TikTok video details in practice
    video_id = "sample_tiktok_12345"
    author_id = "tiktok_user_67890"
    author_username = "health_influencer"
    description = "Natural remedy for common cold #health #natural"

    # Check if the sample file exists (this is just for the example)
    if os.path.exists(sample_video_path):
        # Save the video to the project's storage
        saved_path = collector.save_video(
            sample_video_path, video_id, author_id, author_username, description
        )

        if saved_path:
            # Record manual observations
            collector.record_manual_observation(
                video_id,
                {
                    "content_type": "health advice",
                    "claims_made": ["natural remedies cure common cold"],
                    "visual_elements": [
                        "person showing herbs",
                        "testimonial text overlay",
                    ],
                    "potential_misinformation": True,
                    "notes": "Makes unverified health claims without scientific evidence",
                },
            )
    else:
        logger.warning("This is just an example. No actual video file was processed.")
        logger.info(
            "In real usage, you would download a TikTok video and provide its path"
        )


if __name__ == "__main__":
    # This is just an example and won't run with the placeholder path
    logger.info("This is an example script for manual TikTok collection")
    logger.info("In real usage, you would implement a proper workflow for your team")
