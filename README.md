# MIND Contextual Multi-Armed Bandit Evaluation

This project implements and evaluates contextual Multi-Armed Bandit (MAB) algorithms for news article recommendation using the Microsoft News Dataset (MIND). The focus is on comparing different bandit strategies in an offline evaluation setting with chronological ordering.

## Overview

The project addresses the challenge of personalized news recommendation by treating it as a bandit problem. Instead of recommending individual news articles (51,282 unique items), we use news categories as "arms" (16 categories total), making the problem computationally feasible.

### Key Components

- **Data Source**: MINDsmall_train dataset containing user behaviors, news articles, and entity embeddings
- **Algorithms Evaluated**:
  - ε-Greedy (non-decaying and decaying variants)
  - LinUCB (Linear Upper Confidence Bound)
  - Thompson Sampling
  - Random Baseline
- **Evaluation Metrics**: Cumulative regret and cumulative rewards over time
- **Contextual Features**: Semantic similarity, user history length, category match ratios

## Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning repositories if needed)

### Data Download
1. Download the MIND dataset from the official repository:
   ```bash
   # Visit https://msnews.github.io/ and download MINDsmall_train.zip
   # Extract the contents to a folder named MINDsmall_train/
   ```

2. Place the following files in the `MINDsmall_train/` directory:
   - `behaviors.tsv`
   - `news.tsv`
   - `entity_embedding.vec`
   - `relation_embedding.vec`

### Environment Setup
1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv mind_mab_env
   source mind_mab_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

### Alternative: Using requirements.txt
If a `requirements.txt` file is provided:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the MIND (Microsoft News Dataset) small training set:
- `behaviors.tsv`: User interaction data (impressions and clicks)
- `news.tsv`: News articles with metadata, categories, and entity information
- `entity_embedding.vec`: Pre-trained entity embeddings
- `relation_embedding.vec`: Entity relation embeddings

## Methodology

### Problem Formulation
- **Arms**: 16 news categories (lifestyle, health, news, sports, etc.)
- **Context**: User features including reading history and semantic preferences
- **Reward**: Binary click/no-click (1/0)
- **Challenge**: Offline evaluation with counterfactual estimation

### Feature Engineering
- `semantic_similarity_category`: Cosine similarity between user reading history and category profiles using Wikidata embeddings
- `history_length`: Number of articles user has clicked previously
- `cat_match_ratio`: Ratio of historical clicks matching the current category
- `category_shown`: One-hot encoded category features

### Reward Modeling
A logistic regression model with interaction terms estimates the "true" expected reward for each user-category combination. This serves as the ground truth for regret calculation.

### Bandit Algorithms
- **ε-Greedy**: Balances exploration (random selection) and exploitation (greedy choice)
- **LinUCB**: Uses linear models to estimate rewards with confidence bounds
- **Thompson Sampling**: Probabilistic approach using Beta distributions for arm selection

## Results

The evaluation compares algorithms based on cumulative regret over time. Key findings include performance comparisons across different exploration strategies and the impact of contextual information.

## Files Structure

```
├── BT5153 contextual MABs notebook.ipynb  # Main analysis notebook
├── simulation_cumulative_regret.csv                   # Regret results
├── simulation_cumulative_rewards.csv                  # Reward results
├── images/
│   ├── events_df_plots.png                           # Data exploration plots
│   └── featured_df_plots.png                         # Feature analysis plots
└── MINDsmall_train/
    ├── behaviors.tsv                                 # User behavior data
    ├── news.tsv                                      # News articles
    ├── entity_embedding.vec                          # Entity embeddings
    └── relation_embedding.vec                        # Relation embeddings
```

## Limitations

1. **Category-level Aggregation**: Using categories instead of individual articles reduces granularity but enables feasible computation
2. **Cold Start Bias**: Dataset filtered to users with ≥150 impressions to avoid initialization issues, potentially biasing results
3. **Offline Evaluation**: Relies on logged data and counterfactual estimation rather than live A/B testing

## Dependencies

See the [Setup](#setup) section for detailed installation instructions. Required packages include:

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Usage

1. Ensure MINDsmall_train data is in the correct directory
2. Run the Jupyter notebook sequentially
3. Results will be saved as CSV files and plots in the images/ directory

## References

- Microsoft News Dataset (MIND): https://msnews.github.io/
- Multi-Armed Bandit literature (ε-Greedy, LinUCB, Thompson Sampling)