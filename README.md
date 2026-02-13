[![DOI](https://zenodo.org/badge/1157131514.svg)](https://doi.org/10.5281/zenodo.18633027)

# The Far-Right Telegram Ecosystem Dataset (1025 Channels, 5.7M+ Posts)

This repository contains public Telegram data of 1025 far-right groups used to apply BERT-based topic modeling. The dataset contains over 5.7 million public posts posted between 2019-2024. 

The repo includes processed data, BERT model example outputs, and other materials used in the corresponding paper (to be published).

The repository is organized into several directories containing various components of the project. Below is a detailed description of the repository structure, usage, and how to cite this work.

## Repository Structure

- additional-figures/: Contains visualizations of the model's output in HTML format, used to guide research.
- data/: Contains the processed data, which includes chunked-up CSV files ready for analysis.
- docs/: Contains: a CSV file with all channel names of the dataset, a CSV file with Active Club channel names used to filter dataset and process with BERTopic.
- figures/: Contains figures used in the associated paper in JPG format.
- model-output/: Contains the output of the topic model in CSV format, including topic labels and related data.
- ./: For easy exploration of the dataset's themes or reproducability of the associated paper: run-model.py runs BERTopic on a specific subset of channels in the data in /data and stores the output in /ouput.

## Dataset

The dataset used in this project is obtained from public Telegram channels and groups. The data consists of messages and interactions, which were processed into CSV format. 

### Data Description:

- The processed data is stored in the `data/` directory. It includes multiple CSV files that have been chunked to fit into BERT-based topic models.
- Each CSV file contains columns for message text, date, user (if available), and metadata.
  
We ensured respect for Telegram's [Terms of Service](https://telegram.org/tos) and data privacy laws (e.g., GDPR, CCPA) when working with this dataset.

### Model Settings

The BERT-based topic modeling pipeline includes the following settings:

1. **CountVectorizer**:
    - **stop_words**: 'english' (removes common English stopwords)
    - **min_df**: 2 (ignores terms that appear in fewer than 2 documents)
    - **ngram_range**: (1, 2) (uses unigrams and bigrams)

2. **HDBSCAN** (for clustering topics):
    - **min_cluster_size**: 150 (minimum size of each cluster)
    - **metric**: 'euclidean' (distance metric for clustering)
    - **cluster_selection_method**: 'eom' (used for selecting clusters)
    - **prediction_data**: True (enables prediction on new data)

3. **UMAP** (for dimensionality reduction):
    - **n_neighbors**: 15 (number of neighboring points used for manifold approximation)
    - **n_components**: 5 (reduces the data to 5 components for the clustering)
    - **min_dist**: 0.0 (controls how tightly UMAP packs points together)
    - **metric**: 'cosine' (distance metric for UMAP)

4. **BERTopic** (the main topic model):
    - **embedding_model**: Your BERT embedding model (e.g., `bert-base-uncased`)
    - **umap_model**: UMAP model used for dimensionality reduction
    - **vectorizer_model**: CountVectorizer model used for tokenization
    - **top_n_words**: 20 (number of top words to display for each topic)
    - **verbose**: True (provides detailed output during the topic modeling process)

5. **UMAP for Reduced Embeddings**:
    - **n_neighbors**: 10 (number of neighboring points for the reduced space)
    - **n_components**: 2 (reduces the embeddings to 2 components for visualization)
    - **min_dist**: 0.0 (controls how tightly UMAP packs points together)
    - **metric**: 'cosine' (distance metric for UMAP)

# Example of reduced embeddings visualization

```python
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
```

## Results

The topic modeling results are stored in the `model-output/` folder in CSV format. 

## Figures

The visualizations of the BERT model's output, including topic distributions and other figures used in the associated paper, are stored in the `figures/` folder. Images from Telegram are in JPG format and can be directly used in publications or presentations.

# Data Privacy and Anonymization

This dataset contains public data scraped from public Telegram channels and groups. While all content in the dataset is publicly available, it may contain personal information, including email addresses or contact information that users have voluntarily shared in their public posts.

In accordance with privacy best practices, we have done our best to anonymize the dataset by removing or replacing sensitive information (e.g., email addresses). However, please note that users' publicly shared content is still present and that we may have missed something.

Disclaimer: No private conversations or data from private groups have been included in this dataset. The data is sourced only from publicly accessible Telegram groups/channels.

By using this dataset, you agree to respect the privacy of individuals and adhere to the relevant privacy laws and regulations.

## License

This repository is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset is obtained from public Telegram channels and groups. While we provide the data in this repository, please ensure you follow Telegram's Terms of Service and data privacy laws when using this dataset.

## Citation

If you use this dataset or model in your research, please cite the following:

```bibtex
@misc{nazar2026,
  author = {Shahin Nazar, Thomas F. K. Jorna, Abigail Nieves Delgado, Toine Pieters},
  title = {"The Far-Right Telegram Ecosystem Dataset (1025 Channels, 5.7M+ Posts)"},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18633027},
  url = {https://zenodo.org/records/18633027}
}
```