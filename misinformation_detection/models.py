from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from misinformation_detection.database import Base


class Tweet(Base):
    __tablename__ = "tweets"

    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    author_id = Column(String, nullable=False)
    author_username = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    like_count = Column(Integer, default=0)
    retweet_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    quote_count = Column(Integer, default=0)
    collected_at = Column(DateTime, default=datetime.utcnow)
    has_media = Column(Boolean, default=False)
    
    # Relationships
    media_items = relationship("TwitterMedia", back_populates="tweet")
    analysis_results = relationship("MisinformationAnalysis", back_populates="tweet")


class TwitterMedia(Base):
    __tablename__ = "twitter_media"

    id = Column(String, primary_key=True)
    tweet_id = Column(String, ForeignKey("tweets.id"), nullable=False)
    media_type = Column(String, nullable=False)  # photo, video, etc.
    url = Column(String, nullable=False)
    local_path = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    
    # Relationships
    tweet = relationship("Tweet", back_populates="media_items")


class TikTokVideo(Base):
    __tablename__ = "tiktok_videos"

    id = Column(String, primary_key=True)
    author_id = Column(String, nullable=False)
    author_username = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    local_path = Column(String, nullable=False)
    collected_at = Column(DateTime, default=datetime.utcnow)
    collected_by = Column(String, nullable=False)  # Username of team member who collected it
    
    # Relationships
    analysis_results = relationship("MisinformationAnalysis", back_populates="tiktok_video")


class MisinformationAnalysis(Base):
    __tablename__ = "misinformation_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tweet_id = Column(String, ForeignKey("tweets.id"), nullable=True)
    tiktok_video_id = Column(String, ForeignKey("tiktok_videos.id"), nullable=True)
    text_score = Column(Float, nullable=True)  # Higher score = more likely to be misinformation
    image_score = Column(Float, nullable=True)
    combined_score = Column(Float, nullable=True)
    is_potential_misinformation = Column(Boolean, default=False)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    explanation = Column(Text, nullable=True)
    
    # Relationships
    tweet = relationship("Tweet", back_populates="analysis_results")
    tiktok_video = relationship("TikTokVideo", back_populates="analysis_results")
