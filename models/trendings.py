import re
import pandas as pd
import psycopg2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from translate import translate_to_english  # Import the translation function

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Database connection details
db_config = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'DataEngineer',
    'host': 'localhost',
    'port': 5432
}

# Function to connect to the database
def connect_to_db():
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None, None

# Fetch data from the videos and comments tables
def fetch_data_from_db():
    try:
        conn, cursor = connect_to_db()
        if conn and cursor:
            # Query data from the videos table
            cursor.execute("SELECT * FROM videos;")
            videos = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

            # Query data from the comments table
            cursor.execute("SELECT * FROM comments;")
            comments = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

            print("Data fetched successfully from the database!")
            return videos, comments
    except Exception as e:
        print(f"Error fetching data from the database: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Preprocess the comments
def preprocess(text: str) -> str:
    """
    Preprocesses the input text by replacing usernames and links with placeholders.
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to predict sentiment
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,  # Maximum sequence length for xlm-roberta-base
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    return sentiment_labels[predicted_class]

# Function to preprocess text for topic modeling
def preprocess_for_topic_modeling(text):
    """
    Preprocesses text for topic modeling by cleaning, tokenizing, and lemmatizing.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Function to translate comments to English
def translate_to_english_batch(comments):
    """
    Translates a batch of comments to English.
    """
    translated_comments = []
    for comment in comments:
        if pd.notna(comment):  # Ensure the comment is not None or NaN
            try:
                translated_comment, _ = translate_to_english(comment)
                translated_comments.append(translated_comment if translated_comment else comment)
            except Exception as e:
                print(f"Translation failed for comment: {comment}. Error: {e}")
                translated_comments.append(comment)  # Use the raw comment if translation fails
        else:
            translated_comments.append('')
    return translated_comments

# Function to extract topics using LDA
def extract_topics(comments, num_topics=3, num_words=5):
    """
    Extracts topics from a list of comments using LDA.
    """
    # Preprocess all comments
    processed_comments = [preprocess_for_topic_modeling(comment) for comment in comments]

    # Debug: Print the preprocessed comments
    print("Preprocessed Comments:")
    print(processed_comments)

    # Check if processed_comments is empty
    if not any(processed_comments):
        print("No meaningful terms found after preprocessing. Skipping topic extraction.")
        return []

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(processed_comments)
    corpus = [dictionary.doc2bow(comment) for comment in processed_comments]

    # Apply LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Extract and return the topics
    topics = lda_model.print_topics(num_words=num_words)
    return topics

# Main function to get video topics
def get_video_topics():
    # Fetch data from the database
    videos, comments = fetch_data_from_db()
    if videos is None or comments is None:
        return

    # Ensure video_id is of the same type and trimmed
    videos['video_id'] = videos['video_id'].astype(str).str.strip()
    comments['video_id'] = comments['video_id'].astype(str).str.strip()

    # Aggregate comments by video_id
    comments_agg = comments.groupby('video_id')['comment'].apply(' '.join).reset_index()

    # Merge videos and aggregated comments
    data = pd.merge(videos, comments_agg, on='video_id', how='inner')

    # Sort the DataFrame by created_time (latest first)
    data['created_time'] = pd.to_datetime(data['created_time'])
    data = data.sort_values(by='created_time', ascending=False)

    # Translate comments to English
    data['translated_comment'] = translate_to_english_batch(data['comment'].tolist())

    # Load the tokenizer and model
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Apply sentiment analysis to each video's translated comments
    data['sentiment'] = data['translated_comment'].apply(preprocess).apply(lambda x: predict_sentiment(x, tokenizer, model))

    # Filter for neutral and positive videos
    filtered_data = data[data['sentiment'].isin(['Neutral', 'Positive'])].copy()

    # Extract topics for each video's translated comments
    for index, row in filtered_data.iterrows():
        video_id = row['video_id']
        video_comments = row['translated_comment']

        print(f"\nExtracting topics for video {video_id}...")
        topics = extract_topics([video_comments], num_topics=3, num_words=5)

        # Print the topics
        if topics:
            print(f"Topics for video {video_id}:")
            for topic in topics:
                print(topic)
        else:
            print(f"No topics extracted for video {video_id}.")

# Run the script
if __name__ == "__main__":
    get_video_topics()