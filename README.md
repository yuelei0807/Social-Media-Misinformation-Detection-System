# Social-Media-Misinformation-Detection-System

## Project Overview
This project aims to develop a system that detects and analyzes potential misinformation related to health topics on social media platforms. The system focuses on Twitter as the primary data source, with limited manual TikTok data collection as a secondary source.

## Key Features
- Twitter data collection using free API access
- Manual TikTok data collection tools and workflows
- Text and image processing pipelines
- Misinformation detection using open-source models
- Interactive dashboard for visualization and analysis

## Technology Stack
- Python 3.9+
- PostgreSQL
- Tweepy (Twitter API)
- NLTK, OpenCV, TensorFlow/PyTorch
- Hugging Face Transformers
- Streamlit

## Project Structure

Social-Media-Misinformation-Detection-System/
├── .github/                       # GitHub workflow configs
│   ├── ISSUE_TEMPLATE/            # Issue templates
│   └── workflows/                 # CI/CD workflows
├── alembic/                       # Database migrations
├── config/                        # Configuration files
├── data/                          # Data storage (gitignored)
│   ├── twitter_tweets/            # All tweets
│   ├── twitter_text_only/         # Tweets with only text
│   ├── twitter_with_images/       # Tweets with images
│   ├── twitter_images/            # Twitter images
│   ├── twitter_frames/            # Twitter frames
│   ├── tiktok_videos/             # TikTok videos
│   └── tiktok_frames/             # TikTok frames
├── docs/                          # Documentation
├── misinformation_detection/      # Main package
│   ├── data_collection/           # Collection modules
│   ├── processing/                # Processing modules
│   ├── analysis/                  # Analysis modules
│   ├── visualization/             # Visualization modules
│   └── utils/                     # Utility functions
├── notebooks/                     # Analysis notebooks
├── scripts/                       # Utility scripts
├── tests/                         # Test suite


## Installation
Instructions for setting up the project will be added soon.
