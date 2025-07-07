# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:46:23 2025

@author: Zheng_Wang

To create a weekly historical S&P 500 ticker list from Jan 2022 to Jan 2025, for use as a filter to avoid survivorship bias in social sentiment scraping.

"""


import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# === Step 1: Get all S&P 500 additions/removals from Wikipedia ===
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

# === Step 2: Get the current constituents table ===
tables = pd.read_html(page.text)
current_table = tables[0]
current_sp500 = set(current_table['Symbol'].tolist())

# Create a ticker → name mapping from current table
ticker_to_name = dict(zip(current_table['Symbol'], current_table['Security']))

# === Step 3: Historical changes table ===
changes_table = tables[1]
changes_table.columns = ['date', 'added_tickers', 'added_names', 'removed_tickers', 'removed_names', 'notes']

# Parse change records into a list of (date, ticker, action, name)
change_log = []
for _, row in changes_table.iterrows():
    try:
        date = pd.to_datetime(row['date'])
    except:
        continue
    if isinstance(row['added_tickers'], str):
        added_tickers = [t.strip().upper() for t in row['added_tickers'].split(',')]
        added_names = [n.strip() for n in row['added_names'].split(',')]
        for t, n in zip(added_tickers, added_names):
            change_log.append((date, t, 'ADD', n))
            ticker_to_name[t] = n
    if isinstance(row['removed_tickers'], str):
        removed_tickers = [t.strip().upper() for t in row['removed_tickers'].split(',')]
        removed_names = [n.strip() for n in row['removed_names'].split(',')]
        for t, n in zip(removed_tickers, removed_names):
            change_log.append((date, t, 'REMOVE', n))
            ticker_to_name[t] = n

# Sort chronologically
change_log.sort(reverse=True)

# === Step 4: Rebuild weekly constituent snapshots from Jan 2022 to Jan 2025 ===
start_date = datetime(2022, 1, 3)
end_date = datetime(2025, 1, 6)

constituents = set(current_sp500)
weekly_snapshots = []
snapshot_date = end_date

while snapshot_date >= start_date:
    # Apply changes in reverse (to backtrack from current list)
    while change_log and change_log[0][0] > snapshot_date:
        _, ticker, action, _ = change_log.pop(0)
        if action == 'ADD':
            constituents.discard(ticker)
        elif action == 'REMOVE':
            constituents.add(ticker)

    snapshot = {
        'week_start': snapshot_date.strftime('%Y-%m-%d'),
        'tickers': sorted(constituents),
        'names': [ticker_to_name.get(t, "NA") for t in sorted(constituents)]
    }
    weekly_snapshots.append(snapshot)
    snapshot_date -= timedelta(days=7)

# === Step 5: Save to CSV ===
df_snapshots = pd.DataFrame(weekly_snapshots)
df_snapshots.to_csv("historical_sp500_by_week.csv", index=False)
print("✅ Saved: historical_sp500_by_week.csv")


