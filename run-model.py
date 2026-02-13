# Python 3.x

import pandas as pd
import csv
import glob
import collections
import re

clubnames = r"./docs/network-channel-names.csv"
print(r"Reading clubnames...")
clubnames_df = pd.read_csv(clubnames)
clubnames_list = clubnames_df["clubname"].dropna().tolist()

docs = []

datasets_chunked = glob.glob(r"./data/processed_part_*.csv")

# collect all unique channel names
channelnames = []

# Process each file to find channel names
for file in datasets_chunked:
    df = pd.read_csv(file, usecols=["channel_name"])
    channelnames.extend(df["channel_name"].dropna().unique())

# convert to a Dataframe with unique values
channelname_df = pd.DataFrame({"channel_name":list(set(channelnames))})
save_path = "./output/allchannelnames.csv"
channelname_df.to_csv(save_path, index=False, encoding="utf-8")

# Convert channel names to list for regex filtering
channelnames_list = channelname_df["channel_name"].tolist()
channelnames_pattern = r'\b(?:' + '|'.join(map(re.escape, channelnames_list)) + r')\b'

docs = []  # Use a list to store unique messages
unique_docs = set()  # Track seen messages for deduplication

for dataset in datasets_chunked:
    print(f"Processing {dataset}")
    try:
        # Read file in chunks to avoid memory issues
        df_iter = pd.read_csv(dataset, usecols=["channel_name", "cleaned_message"],
                              sep=None, engine="python", encoding="utf-8", chunksize=10000)

        for chunk in df_iter:
            # Filter rows where channel_name is in clubnames
            filtered_chunk = chunk[chunk["channel_name"].isin(clubnames_df["clubname"])].copy()
            
            # Drop duplicate cleaned_message entries
            filtered_chunk = filtered_chunk.drop_duplicates(subset=["cleaned_message"])

            # Remove words in cleaned_message that match any entry in clubnames_list
            filtered_chunk.loc[:, "cleaned_message"] = filtered_chunk["cleaned_message"].str.replace(channelnames_pattern, '', regex=True)

            # Add unique cleaned_message entries to docs while maintaining order
            for msg in filtered_chunk["cleaned_message"].dropna():
                if msg not in unique_docs:
                    unique_docs.add(msg)
                    docs.append(msg)

    except Exception as e:
        print(f"Error processing {dataset}: {e}")

print(f"Total unique documents collected: {len(docs)}")

# total document duplicates need removal here, I did this manually in excel.

from bertopic import BERTopic

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = embedding_model.encode(docs, show_progress_bar=True)

from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    top_n_words=20,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)

reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

print("Docs length:", len(docs))
print("Topics length:", len(topics))
print("Model topics length:", len(topic_model.topics_))

df = pd.DataFrame({"topic": topics, "document": docs})

save_path_df = "./output/topic-model-results.csv"
df.to_csv(save_path_df, index=False, encoding="utf-8")