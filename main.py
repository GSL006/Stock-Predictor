import streamlit as st
import pandas as pd
import os
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
import yfinance as yf
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Stock Predictor 📈")

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

API_KEY = 'ENTER_YOUR_API_KEY'
API_SECRET = 'ENTER_YOUR_API_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, base_url='https://paper-api.alpaca.markets')
assets = api.list_assets()
ticker_symbols = [asset.symbol for asset in assets if asset.tradable]

# css and html section
st.markdown(
    """
    <style>
    .title {
        font-size: 58px;
        text-align: center;
        color: white;
        font-family: monospace;
    }
    .main {
        font-family: monospace;
        background-image: url('https://www.acquisition-international.com/wp-content/uploads/2020/12/marketing.jpg');
        background-size: cover;
        color: white;
    }
    .stButton {
        padding-top: 70px;
        text-align: center; 
    }
    .stButton button {
        text-align: center;
        vertical-align: middle;
        appearance: none;
        background-color: #FFFFFF;
        border-width: 0;
        box-sizing: border-box;
        color: #000000;
        cursor: pointer;
        display: inline-block;
        font-family: Clarkson,Helvetica,sans-serif;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0;
        line-height: 1em;
        margin: 0 auto;
        opacity: 1;
        outline: 0;
        padding: 1.5em 2.2em;
        position: relative;
        text-decoration: none;
        text-rendering: geometricprecision;
        text-transform: uppercase;
        transition: opacity 300ms cubic-bezier(.694, 0, 0.335, 1),background-color 100ms cubic-bezier(.694, 0, 0.335, 1),color 100ms cubic-bezier(.694, 0, 0.335, 1);
        user-select: none;
        -webkit-user-select: none;
        touch-action: manipulation;
        white-space: nowrap;
        z-index: 1;
    }
    
    .stButton button:hover {
        color: #FFFFFF;
    }

    .stButton button:before {
        border-radius : 5px;
        animation: opacityFallbackOut .5s step-end forwards;
        backface-visibility: hidden;
        background-color: #3CB043;
        clip-path: polygon(-1% 0, 0 0, -25% 100%, -1% 100%);
        content: "";
        height: 100%;
        left: 0;
        position: absolute;
        top: 0;
        transform: translateZ(0);
        transition: clip-path .5s cubic-bezier(.165, 0.84, 0.44, 1), -webkit-clip-path .5s cubic-bezier(.165, 0.84, 0.44, 1);
        width: 100%;
        z-index: -1;
    }

    .stButton button:hover:before {
        animation: opacityFallbackIn 0s step-start forwards;
        clip-path: polygon(0 0, 101% 0, 101% 101%, 0 101%);
    }

    .stButton button span {
        z-index: 1;
        position: relative;
    }
    .title{
        padding-bottom: 60px;
    }
    input, .st-TextInput label, .st-DateInput label, .st-DateInput input{
        font-family: monospace !important;
        color: #000000;
    }
    
     p {
        font-family: monospace;
    }
    </style>
    
    <div class="title">STOCK PREDICTOR 📈</div>
    
    """,
    unsafe_allow_html=True
)

# Input section

ticker = st.selectbox('Enter Stock Ticker:', ticker_symbols)


end_date = st.date_input('Select Date:', datetime.now())
start_date = end_date - timedelta(days=14)

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')


if st.button('Generate Prediction'):
    with st.spinner('Generating prediction...'):
        # Fetching news csv
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

        def fetch_news_data(api, stocks, start_date, end_date):
            news_data = []
            for stock in stocks:
                news = api.get_news(stock, start=start_date, end=end_date, limit=250*12*3)
                for item in news:
                    news_data.append({
                        'ticker': stock,
                        'headline': item.headline,
                        'summary': item.summary,
                        'published_at': item.created_at
                    })
            return pd.DataFrame(news_data)

        df = fetch_news_data(api, [ticker], start_date_str, end_date_str)

        sia = SentimentIntensityAnalyzer()

        def analyze_sentiment(text):
            sentiment = sia.polarity_scores(text)
            if sentiment['compound'] >= 0.05:
                return 'positive', sentiment['compound']
            elif sentiment['compound'] <= -0.05:
                return 'negative', sentiment['compound']
            else:
                return 'neutral', sentiment['compound']

        df['sentiment'], df['sentiment_score'] = zip(*df.apply(lambda row: analyze_sentiment(row['headline'] + ' ' + row['summary']), axis=1))

        def fetch_closest_prices(symbol, news_time):
            try:
                start_date = (news_time - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (news_time + timedelta(days=30)).strftime('%Y-%m-%d')
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)

                if hist_data.empty:
                    return None, None

                before_news = hist_data[hist_data.index < news_time]
                after_news = hist_data[hist_data.index > news_time]

                if not before_news.empty:
                    price_before = before_news['Close'].iloc[-1]
                else:
                    price_before = hist_data['Close'].iloc[0]

                if not after_news.empty:
                    price_after = after_news['Close'].iloc[0]
                else:
                    price_after = hist_data['Close'].iloc[-1]

                return price_before, price_after

            except Exception as e:
                print(f"Error fetching historical prices for {symbol}: {str(e)}")
                return None, None

        def get_prices_around_news_yfinance(row):
            symbol = row['ticker']
            published_at = row['published_at']
            news_time = pd.to_datetime(published_at).tz_convert('America/New_York')

            price_before, price_after = fetch_closest_prices(symbol, news_time)

            if price_before is None or price_after is None:
                print(f"Using fallback mechanism for {symbol} around {news_time}")
                price_before, price_after = fetch_closest_prices(symbol, news_time)

            return price_before, price_after

        if not df.empty:
            df[['price_before', 'price_after']] = df.apply(get_prices_around_news_yfinance, axis=1, result_type='expand')
        else:
            df = pd.DataFrame(columns=['ticker', 'headline', 'summary', 'published_at', 'sentiment', 'sentiment_score', 'price_before', 'price_after'])

        df['published_at_date'] = pd.to_datetime(df['published_at']).dt.date

        daily_sentiment = df.groupby(['ticker', 'published_at_date'])['sentiment_score'].mean().reset_index()
        daily_sentiment.rename(columns={'sentiment_score': 'daily_sentiment'}, inplace=True)

        df = pd.merge(df, daily_sentiment, left_on=['ticker', 'published_at_date'], right_on=['ticker', 'published_at_date'], how='left')

        # Topic modelling
        
        stop_words = set(nltk.corpus.stopwords.words('english'))
        lemmatizer = nltk.WordNetLemmatizer()

        def preprocess(text):
            tokens = nltk.word_tokenize(text.lower())
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
            return tokens

        def choose_summary_or_headline(row):
            text = row['summary'] if pd.notnull(row['summary']) and row['summary'].strip() != '' else row['headline']
            return preprocess(text)

        df['processed'] = df.apply(choose_summary_or_headline, axis=1)

        texts = df['processed'].tolist()
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

        topic_labels = {
            0: 'Customer',
            1: 'Finance',
            2: 'Internal Business Product',
            3: 'Stockholder',
            4: 'Macroeconomics'
        }

        def get_topic(text):
            bow = dictionary.doc2bow(text)
            topic_dist = lda_model[bow]
            topic = sorted(topic_dist, key=lambda x: x[1], reverse=True)[0][0]
            return topic_labels[topic]

        df['topic'] = df['processed'].apply(get_topic)

        df = df.drop(columns=['processed'])

        news_csv_file = os.path.join(os.getcwd(), 'stock_news_data.csv')
        df.to_csv(news_csv_file, index=False)

        # Fetch historical csv
        stock_data = yf.download(ticker, start=start_date_str, end=end_date_str)
        stock_data = stock_data.round(2)
        all_business_days = pd.bdate_range(start=start_date_str, end=end_date_str)
        stock_data = stock_data.reindex(all_business_days)
        stock_data['Ticker'] = ticker
        stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], how='all')
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={'index': 'Date'}, inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])

        # Fetch Close_t-1 to Close_t-7 data
        def get_previous_closes(symbol, end_date, num_days=7):
            ticker = yf.Ticker(symbol)
            closes = []
            current_date = end_date
            while len(closes) < num_days:
                current_date -= timedelta(days=1)
                try:
                    history = ticker.history(start=current_date.strftime('%Y-%m-%d'), end=(current_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                    if not history.empty:
                        closes.append(history['Close'].iloc[0])
                except Exception as e:
                    print(f"Error fetching historical prices for {symbol}: {str(e)}")
            return closes

        for index, row in stock_data.iterrows():
            closes = get_previous_closes(ticker, row['Date'])
            for i, close in enumerate(closes):
                stock_data.at[index, f'Close_t-{i+1}'] = close

        csv_file = f'historical_data.csv'
        stock_data.to_csv(csv_file, index=False)

        historical_csv_file = 'combined_historical_stock_data.csv'
        stock_data.to_csv(historical_csv_file, index=False)

        # Combining both into one final csv
        news_file = news_csv_file
        prices_file = historical_csv_file

        df_news = pd.read_csv(news_file)
        df_prices = pd.read_csv(prices_file)

        df_news['published_at_date'] = pd.to_datetime(df_news['published_at_date'])
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])

        combined_df = pd.merge(df_news, df_prices, left_on=['ticker', 'published_at_date'], right_on=['Ticker', 'Date'], how='inner')
        
        combined_df = combined_df[['Ticker', 'headline', 'summary', 'published_at', 'published_at_date', 'sentiment', 'sentiment_score',
                                'price_before', 'price_after', 'daily_sentiment', 'topic', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                                'Close_t-1', 'Close_t-2', 'Close_t-3', 'Close_t-4', 'Close_t-5', 'Close_t-6', 'Close_t-7']]
        combined_csv_file = 'combined_data.csv'
        combined_df.to_csv(combined_csv_file, index=False)

        # Topic one hot encoding       

        df = pd.read_csv(r'ENTER_VALID_PATH')

        file_path = r'ENTER_VALID_PATH'

        topic_labels = {
            0: 'Customer',
            1: 'Finance',
            2: 'Internal Business Product',
            3: 'Stockholder',
            4: 'Macroeconomics'
        }
        topic_encoded = pd.get_dummies(df['topic']).rename(columns=topic_labels)

        for topic_label in topic_labels.values():
            if topic_label not in topic_encoded.columns:
                topic_encoded[topic_label] = 0

        df_encoded = pd.concat([df, topic_encoded], axis=1)

        df_encoded.drop('topic', axis=1, inplace=True)

        df_encoded.to_csv(file_path, index=False) 
        
        
        
        # Loading the model and the dataset 
        
        model_path = r'stock_price_prediction_model.h5'
        model = load_model(model_path)    
        
        
        df = pd.read_csv(r'C:\STUFF!!!\Uni-Projects\\OWN\ML\combined_data.csv') 
        df['published_at_date'] = pd.to_datetime(df['published_at_date']).dt.tz_localize(None)
        df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)      

        latest_published_at_date = df['published_at_date'].max()
        latest_date_historical= df_prices['Date'].max()
        df_prices = df_prices.iloc[::-1].reset_index(drop=True)

        # Features
        price_feature = [
            df_prices.loc[df_prices['Date'] == latest_date_historical,'Open'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'High'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Low'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-1'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-2'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-3'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-4'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-5'].values[0],
            df_prices.loc[df_prices['Date'] == latest_date_historical, 'Close_t-6'].values[0]
        ]

        sentiment_feature = [
            df.loc[df['published_at_date'] == latest_published_at_date, 'price_before'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'price_after'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'sentiment_score'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'Customer'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'Finance'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'Internal Business Product'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'Stockholder'].values[0],
            df.loc[df['published_at_date'] == latest_published_at_date, 'Macroeconomics'].values[0]
        ]

        def get_sentiment_between(d1, d2, df):
            mask = (df['published_at'] > d1) & (df['published_at'] <= d2)
            sentiments = df.loc[mask, 'daily_sentiment']
            return sentiments.mean() if not sentiments.empty else 0
        
        # Gather sentiments for Close_t-1 to Close_t-7
        for i in range(1, 8):
            t1 = latest_published_at_date - timedelta(days=i)
            t2 = latest_published_at_date - timedelta(days=i-1)
            if t1 >= df['published_at_date'].min():
                sentiment_between = get_sentiment_between(t1, t2, df)
                sentiment_feature.append(sentiment_between)
            else:
                sentiment_feature.append(0)  

        while len(sentiment_feature) < 10:
            sentiment_feature.append(0)
            
        next_day_feature = price_feature + sentiment_feature

        next_day_feature = np.array(next_day_feature).reshape(1, -1)

        # Predict the closing price for the next day
        next_day_prediction = model.predict(next_day_feature)
        latest_date_historical = latest_date_historical.strftime('%Y-%m-%d')

        st.markdown(
            f'<div style="background-color:black; padding:10px; border-radius:5px; color:white; text-align:center; font-family: monospace;">'
            f'Predicted closing price for the next market day after {latest_date_historical} is: ${next_day_prediction[0][0]:.2f}'
            '</div>',
            unsafe_allow_html=True
        )
        
