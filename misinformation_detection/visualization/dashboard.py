import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import datetime

from misinformation_detection.database import Base
from misinformation_detection.models import (
    Tweet,
    TwitterMedia,
    TikTokVideo,
    MisinformationAnalysis,
)
from config.config import DATABASE_URL


# Initialize database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Misinformation Detection Dashboard", page_icon="🔍", layout="wide"
    )

    # Dashboard title
    st.title("Social Media Misinformation Detection")
    st.markdown("### Health Information Analysis Dashboard")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["Overview", "Twitter Analysis", "TikTok Analysis", "Content Details"],
    )

    # Connect to the database
    db = get_db()

    if page == "Overview":
        display_overview(db)
    elif page == "Twitter Analysis":
        display_twitter_analysis(db)
    elif page == "TikTok Analysis":
        display_tiktok_analysis(db)
    elif page == "Content Details":
        display_content_details(db)


def display_overview(db):
    """Display overview page with summary statistics."""
    st.header("Overview")

    # Get summary statistics
    tweet_count = db.query(func.count(Tweet.id)).scalar()
    tiktok_count = db.query(func.count(TikTokVideo.id)).scalar()
    analysis_count = db.query(func.count(MisinformationAnalysis.id)).scalar()
    misinformation_count = (
        db.query(func.count(MisinformationAnalysis.id))
        .filter(MisinformationAnalysis.is_potential_misinformation == True)
        .scalar()
    )

    # Display summary statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tweets Collected", tweet_count)
    with col2:
        st.metric("TikTok Videos Collected", tiktok_count)
    with col3:
        st.metric("Total Analyses", analysis_count)
    with col4:
        misinformation_percentage = (
            (misinformation_count / analysis_count * 100) if analysis_count > 0 else 0
        )
        st.metric("Potential Misinformation", f"{misinformation_percentage:.1f}%")

    # Recent activity
    st.subheader("Recent Activity")

    # Get recent analyses
    recent_analyses = (
        db.query(MisinformationAnalysis)
        .order_by(MisinformationAnalysis.analysis_date.desc())
        .limit(10)
        .all()
    )

    if recent_analyses:
        # Prepare data for plotting
        analysis_data = []
        for analysis in recent_analyses:
            content_id = (
                analysis.tweet_id if analysis.tweet_id else analysis.tiktok_video_id
            )
            content_type = "Twitter" if analysis.tweet_id else "TikTok"
            analysis_data.append(
                {
                    "ID": content_id,
                    "Type": content_type,
                    "Score": analysis.combined_score,
                    "Date": analysis.analysis_date,
                }
            )

        # Create a DataFrame
        df = pd.DataFrame(analysis_data)

        # Create a plot
        fig = px.bar(
            df,
            x="ID",
            y="Score",
            color="Type",
            title="Recent Content Analysis Scores",
            labels={"Score": "Misinformation Score", "ID": "Content ID"},
            height=400,
        )

        # Add a threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df) - 0.5,
            y0=0.6,
            y1=0.6,
            line=dict(color="red", width=2, dash="dash"),
        )

        # Add annotation for threshold
        fig.add_annotation(
            x=len(df) - 1,
            y=0.62,
            text="Misinformation Threshold (0.6)",
            showarrow=False,
            font=dict(color="red"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display recent analyses as a table
        st.dataframe(df)
    else:
        st.info(
            "No analysis data available. Start collecting and analyzing content to see results here."
        )

    # Health topics distribution
    st.subheader("Health Topics Distribution")

    # This is a placeholder - in a real implementation, you would
    # extract health topics from your analyses and visualize them

    # Example data
    topics_data = {
        "COVID-19": 45,
        "Diet & Nutrition": 32,
        "Supplements": 28,
        "Alternative Medicine": 15,
        "Mental Health": 10,
    }

    # Create a pie chart
    fig = px.pie(
        values=list(topics_data.values()),
        names=list(topics_data.keys()),
        title="Distribution of Health Topics",
        hole=0.4,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Misinformation trend over time
    st.subheader("Misinformation Trend")

    # Example time series data
    # In a real implementation, you would query this from your database
    dates = [
        datetime.datetime.now() - datetime.timedelta(days=x) for x in range(14, -1, -1)
    ]
    scores = [
        0.3,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.6,
        0.55,
        0.5,
        0.45,
        0.5,
        0.55,
        0.6,
    ]

    trend_data = pd.DataFrame({"Date": dates, "Average Score": scores})

    # Create a line chart
    fig = px.line(
        trend_data,
        x="Date",
        y="Average Score",
        title="Average Misinformation Score Trend (Last 15 Days)",
        markers=True,
    )

    # Add a threshold line
    fig.add_shape(
        type="line",
        x0=trend_data["Date"].min(),
        x1=trend_data["Date"].max(),
        y0=0.6,
        y1=0.6,
        line=dict(color="red", width=2, dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True)


def display_twitter_analysis(db):
    """Display Twitter analysis page."""
    st.header("Twitter Analysis")

    # Get Twitter statistics
    tweet_count = db.query(func.count(Tweet.id)).scalar()
    media_count = db.query(func.count(TwitterMedia.id)).scalar()
    text_only_count = (
        db.query(func.count(Tweet.id)).filter(Tweet.has_media == False).scalar()
    )
    with_media_count = (
        db.query(func.count(Tweet.id)).filter(Tweet.has_media == True).scalar()
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweets", tweet_count)
    with col2:
        st.metric("Text-Only Tweets", text_only_count)
    with col3:
        st.metric("Tweets with Media", with_media_count)
    with col4:
        st.metric("Media Items", media_count)

    # Get engagement statistics
    engagement_data = db.query(
        func.sum(Tweet.like_count).label("likes"),
        func.sum(Tweet.retweet_count).label("retweets"),
        func.sum(Tweet.reply_count).label("replies"),
        func.sum(Tweet.quote_count).label("quotes"),
    ).first()

    if engagement_data:
        st.subheader("Engagement Metrics")

        engagement_metrics = {
            "Likes": engagement_data.likes or 0,
            "Retweets": engagement_data.retweets or 0,
            "Replies": engagement_data.replies or 0,
            "Quotes": engagement_data.quotes or 0,
        }

        fig = px.bar(
            x=list(engagement_metrics.keys()),
            y=list(engagement_metrics.values()),
            title="Total Engagement by Type",
            labels={"x": "Engagement Type", "y": "Count"},
        )

        st.plotly_chart(fig, use_container_width=True)

    # Top tweets by engagement
    st.subheader("Top Tweets by Engagement")

    top_tweets = (
        db.query(Tweet)
        .order_by(
            (
                Tweet.like_count
                + Tweet.retweet_count
                + Tweet.reply_count
                + Tweet.quote_count
            ).desc()
        )
        .limit(10)
        .all()
    )

    if top_tweets:
        tweet_data = []
        for tweet in top_tweets:
            total_engagement = (
                tweet.like_count
                + tweet.retweet_count
                + tweet.reply_count
                + tweet.quote_count
            )
            tweet_data.append(
                {
                    "ID": tweet.id,
                    "Author": tweet.author_username,
                    "Content": tweet.content[:50] + "..."
                    if len(tweet.content) > 50
                    else tweet.content,
                    "Has Media": tweet.has_media,
                    "Likes": tweet.like_count,
                    "Retweets": tweet.retweet_count,
                    "Total Engagement": total_engagement,
                }
            )

        st.dataframe(pd.DataFrame(tweet_data))
    else:
        st.info("No Twitter data available.")

    # Tweets with potential misinformation
    st.subheader("Tweets with Potential Misinformation")

    misinformation_tweets = (
        db.query(Tweet)
        .join(MisinformationAnalysis, Tweet.id == MisinformationAnalysis.tweet_id)
        .filter(MisinformationAnalysis.is_potential_misinformation == True)
        .order_by(MisinformationAnalysis.combined_score.desc())
        .limit(10)
        .all()
    )

    if misinformation_tweets:
        misinformation_data = []
        for tweet in misinformation_tweets:
            analysis = (
                db.query(MisinformationAnalysis)
                .filter(MisinformationAnalysis.tweet_id == tweet.id)
                .first()
            )

            misinformation_data.append(
                {
                    "ID": tweet.id,
                    "Author": tweet.author_username,
                    "Content": tweet.content[:50] + "..."
                    if len(tweet.content) > 50
                    else tweet.content,
                    "Has Media": tweet.has_media,
                    "Score": analysis.combined_score if analysis else 0,
                    "Engagement": tweet.like_count
                    + tweet.retweet_count
                    + tweet.reply_count
                    + tweet.quote_count,
                }
            )

        # Create a DataFrame
        df = pd.DataFrame(misinformation_data)

        # Create a scatter plot of engagement vs. misinformation score
        fig = px.scatter(
            df,
            x="Score",
            y="Engagement",
            size="Engagement",
            hover_name="Author",
            hover_data=["Content"],
            title="Engagement vs. Misinformation Score",
            labels={"Score": "Misinformation Score", "Engagement": "Total Engagement"},
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display as a table
        st.dataframe(df)
    else:
        st.info("No tweets with potential misinformation detected.")


def display_tiktok_analysis(db):
    """Display TikTok analysis page."""
    st.header("TikTok Analysis")

    # Get TikTok statistics
    tiktok_count = db.query(func.count(TikTokVideo.id)).scalar()

    st.metric("Total TikTok Videos", tiktok_count)

    # Display collector statistics
    st.subheader("Collection by Team Member")

    collector_stats = (
        db.query(TikTokVideo.collected_by, func.count(TikTokVideo.id).label("count"))
        .group_by(TikTokVideo.collected_by)
        .all()
    )

    if collector_stats:
        collector_data = {stat.collected_by: stat.count for stat in collector_stats}

        fig = px.pie(
            values=list(collector_data.values()),
            names=list(collector_data.keys()),
            title="Videos Collected by Team Member",
        )

        st.plotly_chart(fig, use_container_width=True)

    # TikTok videos with potential misinformation
    st.subheader("TikTok Videos with Potential Misinformation")

    misinformation_tiktoks = (
        db.query(TikTokVideo)
        .join(
            MisinformationAnalysis,
            TikTokVideo.id == MisinformationAnalysis.tiktok_video_id,
        )
        .filter(MisinformationAnalysis.is_potential_misinformation == True)
        .order_by(MisinformationAnalysis.combined_score.desc())
        .limit(10)
        .all()
    )

    if misinformation_tiktoks:
        misinformation_data = []
        for video in misinformation_tiktoks:
            analysis = (
                db.query(MisinformationAnalysis)
                .filter(MisinformationAnalysis.tiktok_video_id == video.id)
                .first()
            )

            misinformation_data.append(
                {
                    "ID": video.id,
                    "Author": video.author_username,
                    "Description": video.description[:50] + "..."
                    if video.description and len(video.description) > 50
                    else video.description,
                    "Score": analysis.combined_score if analysis else 0,
                    "Collected By": video.collected_by,
                    "Collection Date": video.collected_at.strftime("%Y-%m-%d"),
                }
            )

        # Display as a table
        st.dataframe(pd.DataFrame(misinformation_data))
    else:
        st.info("No TikTok videos with potential misinformation detected.")

    # Collection over time
    st.subheader("TikTok Collection Over Time")

    # Example time series data
    # In a real implementation, you would query this from your database
    collection_dates = [
        datetime.datetime.now() - datetime.timedelta(days=x) for x in range(14, -1, -1)
    ]
    collection_counts = [0, 1, 2, 0, 1, 0, 0, 3, 1, 0, 2, 1, 0, 0, 1]

    collection_data = pd.DataFrame(
        {"Date": collection_dates, "Videos Collected": collection_counts}
    )

    # Create a bar chart
    fig = px.bar(
        collection_data,
        x="Date",
        y="Videos Collected",
        title="TikTok Videos Collected Over Time (Last 15 Days)",
    )

    st.plotly_chart(fig, use_container_width=True)


def display_content_details(db):
    """Display content details page."""
    st.header("Content Details")

    # Select content type
    content_type = st.radio("Select Content Type", ["Twitter", "TikTok"])

    if content_type == "Twitter":
        # Get all tweets
        tweets = db.query(Tweet).all()

        if tweets:
            # Create a selectbox for choosing a tweet
            tweet_options = [
                f"{tweet.author_username}: {tweet.content[:30]}..." for tweet in tweets
            ]
            selected_index = st.selectbox(
                "Select a Tweet",
                range(len(tweet_options)),
                format_func=lambda x: tweet_options[x],
            )

            if selected_index is not None:
                selected_tweet = tweets[selected_index]

                # Display tweet details
                st.subheader("Tweet Details")
                st.write(f"**Author:** @{selected_tweet.author_username}")
                st.write(f"**Created at:** {selected_tweet.created_at}")
                st.write(f"**Content:**")
                st.write(selected_tweet.content)
                st.write(
                    f"**Has Media:** {'Yes' if selected_tweet.has_media else 'No'}"
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Likes", selected_tweet.like_count)
                with col2:
                    st.metric("Retweets", selected_tweet.retweet_count)
                with col3:
                    st.metric("Replies", selected_tweet.reply_count)
                with col4:
                    st.metric("Quotes", selected_tweet.quote_count)

                # Get media items
                media_items = (
                    db.query(TwitterMedia)
                    .filter(TwitterMedia.tweet_id == selected_tweet.id)
                    .all()
                )

                if media_items:
                    st.subheader("Media Items")

                    for media in media_items:
                        st.write(f"**Type:** {media.media_type}")
                        st.write(f"**URL:** {media.url}")
                        if media.local_path and st.button(f"View Media {media.id}"):
                            # This would display the media in a real implementation
                            st.info(
                                f"Media would be displayed here from: {media.local_path}"
                            )

                # Get misinformation analysis
                analysis = (
                    db.query(MisinformationAnalysis)
                    .filter(MisinformationAnalysis.tweet_id == selected_tweet.id)
                    .first()
                )

                if analysis:
                    st.subheader("Misinformation Analysis")

                    # Create a gauge chart for the score
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=analysis.combined_score,
                            title={"text": "Misinformation Score"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 0.4], "color": "green"},
                                    {"range": [0.4, 0.6], "color": "yellow"},
                                    {"range": [0.6, 1], "color": "red"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 0.6,
                                },
                            },
                        )
                    )

                    st.plotly_chart(fig)

                    st.write("**Analysis Date:** ", analysis.analysis_date)
                    st.write(
                        "**Is Potential Misinformation:** ",
                        "Yes" if analysis.is_potential_misinformation else "No",
                    )

                    if analysis.explanation:
                        st.subheader("Explanation")
                        st.write(analysis.explanation)
                else:
                    st.info("No misinformation analysis available for this tweet.")

        else:
            st.info("No Twitter data available.")

    else:  # TikTok
        # Get all TikTok videos
        tiktok_videos = db.query(TikTokVideo).all()

        if tiktok_videos:
            # Create a selectbox for choosing a video
            tiktok_options = [
                f"{video.author_username}: {video.description[:30]}..."
                if video.description
                else f"{video.author_username}: [No description]"
                for video in tiktok_videos
            ]
            selected_index = st.selectbox(
                "Select a TikTok Video",
                range(len(tiktok_options)),
                format_func=lambda x: tiktok_options[x],
            )

            if selected_index is not None:
                selected_video = tiktok_videos[selected_index]

                # Display video details
                st.subheader("TikTok Video Details")
                st.write(f"**Author:** @{selected_video.author_username}")
                st.write(f"**Collected at:** {selected_video.collected_at}")
                st.write(f"**Collected by:** {selected_video.collected_by}")

                if selected_video.description:
                    st.write(f"**Description:**")
                    st.write(selected_video.description)

                if selected_video.local_path and st.button("View Video"):
                    # This would display the video in a real implementation
                    st.info(
                        f"Video would be played here from: {selected_video.local_path}"
                    )

                # Get misinformation analysis
                analysis = (
                    db.query(MisinformationAnalysis)
                    .filter(MisinformationAnalysis.tiktok_video_id == selected_video.id)
                    .first()
                )

                if analysis:
                    st.subheader("Misinformation Analysis")

                    # Create a gauge chart for the score
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=analysis.combined_score,
                            title={"text": "Misinformation Score"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 0.4], "color": "green"},
                                    {"range": [0.4, 0.6], "color": "yellow"},
                                    {"range": [0.6, 1], "color": "red"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 0.6,
                                },
                            },
                        )
                    )

                    st.plotly_chart(fig)

                    st.write("**Analysis Date:** ", analysis.analysis_date)
                    st.write(
                        "**Is Potential Misinformation:** ",
                        "Yes" if analysis.is_potential_misinformation else "No",
                    )

                    if analysis.explanation:
                        st.subheader("Explanation")
                        st.write(analysis.explanation)
                else:
                    st.info(
                        "No misinformation analysis available for this TikTok video."
                    )

        else:
            st.info("No TikTok data available.")


if __name__ == "__main__":
    run_dashboard()
