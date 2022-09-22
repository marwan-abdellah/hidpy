# Import libraries
# yfinance offers a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance
# Please check out its official doc for details: https://pypi.org/project/yfinance/
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load historical data in the past 10 years
sp500 = yf.Ticker("^GSPC")
end_date = pd.Timestamp.today()
start_date = end_date - pd.Timedelta(days=10*365)
sp500_history=sp500.history(start=start_date, end=end_date)

# Remove unnecessary columns
sp500_history = sp500_history.drop(columns=['Dividends', 'Stock Splits'])

# Create a new column as Close 200 days moving average
sp500_history['Close_200ma'] = sp500_history['Close'].rolling(200).mean()

# Create a summary statistics table
sp500_history_summary = sp500_history.describe()

sns.relplot(data=sp500_history[['Close', 'Close_200ma']], kind='line', height=3, aspect=2.0)
plt.savefig('chart.png')

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Load the example diamonds dataset
diamonds = sns.load_dataset("diamonds")

# Draw a scatter plot while assigning point colors and sizes to different
# variables in the dataset
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
sns.scatterplot(x="carat", y="price",
                hue="clarity", size="depth",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=diamonds, ax=ax)
plt.savefig('chart2.png')

# 1. Set up multiple variables to store the titles, text within the report
page_title_text='HiDPy Report'
title_text = 'HidPy Report - Report #1'
text = 'Section 0'
prices_text = 'Section 1'
stats_text = 'Section 2'
sec3 = 'Section 3'



# 2. Combine them together using a long f-string
html = f'''
    <html>
        <head>
        <meta name="\viewport\" content=\"width=device-width, initial-scale=1\">
            <title>{page_title_text}</title>
        </head>
        <body>
            <h1>{title_text}</h1>
            <h2>{text}</h2>
            <img src='chart.png' width="700">
            <h2>{prices_text}</h2>
            {sp500_history.tail(3).to_html()}
            <h2>{stats_text}</h2>
            {sp500_history_summary.to_html()}
            <h2>{sec3}</h2>
            <img src='chart2.png' width="700">
        </body>
    </html>
    '''
# 3. Write the html string as an HTML file
with open('hidpy-report-1.html', 'w') as f:
    f.write(html)

    
