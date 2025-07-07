# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 21:36:03 2025

# -*- coding: utf-8 -*-

Historical Reddit Scraper for Weekly Top 5 S&P 500 Tickers (2021-2023)

This script scrapes popular financial subreddits to find the top 5 most
mentioned S&P 500 tickers each week. It dynamically filters tickers to ensure
they were part of the S&P 500 during the specific week of mention.

This version has been refactored for clarity and maintainability.

@author: Zheng_Wang
"""

# Standard library imports
import ast
import json
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import pandas as pd
import praw
import spacy
from spacy.matcher import PhraseMatcher

class Config:
    """Holds all configuration settings for the scraper."""
    # Date range for the historical data scrape
    HISTORICAL_START = datetime(2021, 1, 1)
    HISTORICAL_END = datetime(2024, 1, 1)

    # Number of top tickers to select each week
    TOP_N_WEEKLY = 5

    # Path to the CSV file containing weekly S&P 500 constituents
    # IMPORTANT: Update this path to your local file location
    SP500_CSV_PATH = r"C:\Users\YourUser\path\to\your\historical_sp500_by_week.csv"

    # Subreddits to scrape for ticker mentions
    SUBREDDITS = ["stocks", "investing", "wallstreetbets", "SecurityAnalysis", "StockMarket"]

    # PRAW (Reddit API) settings
    # IMPORTANT: Replace these with your own Reddit API credentials
    REDDIT_CLIENT_ID = "YOUR_CLIENT_ID"
    REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
    REDDIT_USER_AGENT = "TickerSentimentScraper v1.0 by YourUsername"
    REDDIT_USERNAME = "YOUR_REDDIT_USERNAME"
    REDDIT_PASSWORD = "YOUR_REDDIT_PASSWORD"

    # Number of posts to fetch from each scraping method (e.g., top, controversial)
    MAX_POSTS_PER_METHOD = 500

    # Output directory for results
    OUTPUT_DIR = Path("reddit_weekly_data")

# Suppress a common warning from PRAW in non-async environments
warnings.filterwarnings("ignore", message="It appears that you are using PRAW in an asynchronous environment")

def initialize_reddit_client(config):
    """Creates and returns an authenticated PRAW Reddit instance."""
    print("🔐 Authenticating with Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
            username=config.REDDIT_USERNAME,
            password=config.REDDIT_PASSWORD,
            check_for_async=False
        )