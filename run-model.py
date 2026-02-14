# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "bertopic>=0.17.4",
#     "natsort>=8.4.0",
#     "pandas>=3.0.0",
# ]
# ///

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import re
import glob
import pandas as pd
from natsort import natsorted

PREFIX = ">>>"

clubnames = r"./docs/network-channel-names.csv"
print(f"{PREFIX} Reading clubnames from {clubnames}...")
clubnames_df = pd.read_csv(clubnames)
clubnames_list = clubnames_df["clubname"].dropna().tolist()
print(f"{PREFIX} Loaded {len(clubnames_list)} club names.")

datasets_chunked = natsorted(glob.glob(r"./data/processed_part_*.csv"))
print(f"{PREFIX} Found {len(datasets_chunked)} data files (sorted numerically).")

channelnames = []

print(f"{PREFIX} Collecting channel names from data files...")
for file in datasets_chunked:
    df = pd.read_csv(file, usecols=["channel_name"])
    channelnames.extend(df["channel_name"].dropna().unique())

# convert to a dataframe with unique values
channelname_df = pd.DataFrame({"channel_name": list(set(channelnames))})
print(f"{PREFIX} Found {len(channelname_df)} unique channel names.")
save_path = "./output/allchannelnames.csv"
channelname_df.to_csv(save_path, index=False, encoding="utf-8")
print(f"{PREFIX} Saved channel names to {save_path}")

# convert channel names to list for regex filtering
channelnames_list = channelname_df["channel_name"].tolist()
channelnames_pattern = r'\b(?:' + '|'.join(map(re.escape, channelnames_list)) + r')\b'

docs = []
unique_docs = set()

print(f"{PREFIX} Filtering and deduplicating documents...")
total = len(datasets_chunked)
for i, dataset in enumerate(datasets_chunked, 1):
    print(f"{PREFIX}   [{i}/{total}] {os.path.basename(dataset)}")
    try:
        # read file in chunks to avoid memory issues
        df_iter = pd.read_csv(dataset, usecols=["channel_name", "cleaned_message"],
                              sep=None, engine="python", encoding="utf-8", chunksize=10000)

        for chunk in df_iter:
            filtered_chunk = chunk[chunk["channel_name"].isin(clubnames_df["clubname"])].copy()
            filtered_chunk = filtered_chunk.drop_duplicates(subset=["cleaned_message"])
            filtered_chunk.loc[:, "cleaned_message"] = filtered_chunk["cleaned_message"].str.replace(channelnames_pattern, '', regex=True)

            for msg in filtered_chunk["cleaned_message"].dropna():
                if msg not in unique_docs:
                    unique_docs.add(msg)
                    docs.append(msg)

    except Exception as e:
        print(f"{PREFIX}   Error processing {dataset}: {e}")

print(f"{PREFIX} Total unique documents collected: {len(docs)}")

# total document duplicates need removal here, I did this manually in excel.

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

print(f"{PREFIX} Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print(f"{PREFIX} Encoding {len(docs)} documents (this may take a while)...")
embeddings = embedding_model.encode(docs, show_progress_bar=True)
print(f"{PREFIX} Encoding complete.")

print(f"{PREFIX} Initializing UMAP, HDBSCAN, and CountVectorizer...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True, core_dist_n_jobs=1)
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

print(f"{PREFIX} Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=20,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs, embeddings)
n_topics = len(set(topics)) - (1 if -1 in topics else 0)
print(f"{PREFIX} BERTopic fitting complete. Found {n_topics} topics.")

print(f"{PREFIX} Computing reduced embeddings for visualization...")
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
print(f"{PREFIX} Reduced embeddings complete.")

print(f"{PREFIX} Docs: {len(docs)} | Topics: {len(topics)} | Model topics: {len(topic_model.topics_)}")

print(f"{PREFIX} Saving results...")
df = pd.DataFrame({"topic": topics, "document": docs})
save_path_df = "./output/topic-model-results.csv"
df.to_csv(save_path_df, index=False, encoding="utf-8")
print(f"{PREFIX} Results saved to {save_path_df}")
print(f"{PREFIX} Done.")
