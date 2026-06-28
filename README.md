# crypto-social-media-analysis
Analysis of cryptocurrency market trends using social media data, sentiment analysis, network analysis, Python, and interactive dashboards.
# Crypto Social Media Analysis

## Project Overview

This project analyzes the impact of cryptocurrency-focused Telegram channels on the behavior of Iranian investors. It connects Telegram channel activity, post engagement, cryptocurrency market movements, and signal accuracy to understand how social media content amplifies or shapes investor reactions.

The main idea of the project is:

> **Price movements and volatility are the main drivers of crypto investor behavior, while Telegram channels amplify and shape those reactions through timely and action-oriented content.**

## Research Question

**Do Telegram channels influence Iranian crypto investors' behavior?**

The analysis suggests that Telegram channels do influence investor behavior, mainly by increasing attention, engagement, and reaction intensity around market movements, volatility, and trading signals.

## Dataset

The Telegram dataset contains approximately **5,000 records** and **9 fields** collected from cryptocurrency-related Telegram channels.

Market data was retrieved using Yahoo Finance data through Python libraries, including daily OHLC prices and trading volume for major cryptocurrencies.

## Data Cleaning and Preprocessing

The preprocessing stage included:

- Removing invalid IDs and incorrect date values
- Filling missing member counts
- Correcting negative values in views and forwards
- Standardizing date formats
- Preparing Telegram post-level data for engagement analysis
- Retrieving market data through Yahoo Finance instead of manual scraping
- Calculating typical price to smooth intraday price noise

## Key Metrics

### Engagement Rate

Engagement Rate is used to measure user attention and reaction to Telegram content.

The metric was analyzed at four levels:

- **Post level:** engagement for each individual message
- **Channel level:** average engagement across posts in a channel
- **Content type level:** engagement by content category
- **Cryptocurrency level:** engagement for posts related to each cryptocurrency

### Average Abnormal Return (AAR)

AAR was used to evaluate the average price movement after Telegram trading signal posts compared with expected market behavior.

This metric helps assess whether signal posts are aligned with later market movements.

### Hit Ratio

Hit Ratio measures the percentage of signal posts followed by an absolute price movement greater than a predefined threshold. In this project, the threshold was set at **1%**.

## Methodology

The project includes the following analytical steps:

1. Data cleaning and exploratory data analysis
2. Engagement Rate analysis by channel, content type, cryptocurrency, and time
3. Correlation analysis between user activity and price changes
4. Time-based analysis of engagement patterns
5. Market phase analysis using moving averages and turning points
6. Volatility and message volume analysis
7. Event study analysis for trading signal accuracy
8. Channel ranking using AAR and Hit Ratio
9. Dashboard design for KPI monitoring and trend visualization

## Main Findings

- Telegram channels significantly shape investor attention and reactions, especially during volatile market conditions.
- Simple linear correlations between price changes, views, engagement rate, and member count were weak or insignificant.
- Telegram activity appears to be more reactive to major market events than predictive of price movements.
- Signal-related content generated the highest engagement, followed by fundamental content.
- Engagement was highest at the beginning of weekly and monthly cycles.
- Monday showed the highest average engagement rate among weekdays.
- January showed the highest average engagement rate among months.
- Trading volume alone did not explain message count or engagement behavior.
- Market direction and volatility both influenced content patterns across major cryptocurrencies.
- Channel size was not strongly linked to signal accuracy; smaller channels sometimes performed slightly better.
- AAR and Hit Ratio provided a better framework for comparing signal quality across channels.

## Selected Results

### Top Engagement Content

Signal posts generated the highest engagement compared with other content categories.

### Timing Patterns

Engagement tended to peak early in the week and during active market periods.

### Signal Accuracy

Signal accuracy was evaluated using:

- 3-day Average Abnormal Return
- 5-day Average Abnormal Return
- Hit Ratio based on a 1% threshold

Some channels showed stronger consistency in signal performance, while others had high abnormal returns but lower consistency.

## Project Structure

```text
crypto-social-media-analysis/
│
├── data/
│   ├── raw/                 # Raw Telegram and market data, not recommended for public upload
│   ├── processed/           # Cleaned and transformed data
│   └── sample/              # Small anonymized sample data for GitHub
│
├── notebooks/               # Jupyter notebooks for EDA and analysis
│
├── src/                     # Python scripts for cleaning, analysis, and metrics
│
├── dashboard/               # Power BI, Excel, or dashboard screenshots
│
├── reports/                 # Final report and project documentation
│
├── docs/                    # Methodology and metric explanations
│
├── README.md
├── README_FA.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tools and Technologies

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Plotly
- NetworkX
- yfinance
- Jupyter Notebook
- Power BI / Dashboard tools
- GitHub

## Dashboard

The dashboard can be used to monitor:

- Number of Telegram posts
- Engagement Rate by channel and content type
- Message volume by cryptocurrency
- Market phase distribution
- Volatility and activity patterns
- AAR and Hit Ratio ranking
- Top-performing channels

## Limitations

- Buy/sell direction was not available for all signal posts.
- AAR can show alignment with price movements but does not fully prove causality.
- Simple correlations may miss nonlinear and event-driven behavior.
- Telegram channel behavior can be affected by external news, market shocks, and investor sentiment.
- Raw Telegram data may contain privacy-sensitive information and should be anonymized before public release.

## Future Improvements

- Apply nonlinear and multivariate models
- Use event detection around major market shocks
- Add sentiment analysis using Persian NLP models
- Compare Telegram data with Twitter/X, news, or Google Trends
- Build a real-time monitoring dashboard
- Develop a scoring model for Telegram channel reliability

## Author

Sarah Azizi

Project developed as part of a cryptocurrency market and social media behavior analysis portfolio.
