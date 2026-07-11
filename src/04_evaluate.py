import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

DATA = "/home/claude/data"
np.random.seed(42)

books = pd.read_csv(f"{DATA}/books_clean.csv")
ratings = pd.read_csv(f"{DATA}/ratings_clean.csv")

# ---- Train / test split (per-interaction, 80/20) ----
train, test = train_test_split(ratings, test_size=0.2, random_state=42,
                                stratify=None)

isbn_to_idx = {isbn: i for i, isbn in enumerate(books["ISBN"])}
books["content"] = (books["Book-Title"].fillna("") + " " +
                     books["Book-Author"].fillna("") + " " +
                     books["Publisher"].fillna(""))
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(books["content"])
content_sim = cosine_similarity(tfidf_matrix, dense_output=False)

user_ids = train["User-ID"].unique()
user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
global_mean = train["Book-Rating"].mean()

# Mean-center ratings before factorization (raw ratings carry a large constant
# offset that plain SVD does not model well on its own)
row = train["User-ID"].map(user_id_to_idx)
col = train["ISBN"].map(isbn_to_idx)
data = (train["Book-Rating"] - global_mean).values
user_item_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(books)))

svd = TruncatedSVD(n_components=30, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T


def predict_rating(user_id, isbn):
    """Predict rating via CF factors (mean-centered); fall back to global mean for cold-start."""
    if user_id in user_id_to_idx and isbn in isbn_to_idx:
        pred = global_mean + item_factors[isbn_to_idx[isbn]] @ user_factors[user_id_to_idx[user_id]]
        return np.clip(pred, 1, 10)
    return global_mean


def normalize(scores):
    rng = scores.max() - scores.min()
    return (scores - scores.min()) / rng if rng > 0 else scores


def hybrid_topn_for_user(user_id, train_isbns, alpha=0.5, n=10):
    cf_scores = (item_factors @ user_factors[user_id_to_idx[user_id]]
                 if user_id in user_id_to_idx else np.zeros(len(books)))
    user_ratings_map = dict(zip(train[train["User-ID"] == user_id]["ISBN"],
                                 train[train["User-ID"] == user_id]["Book-Rating"]))
    cb_scores = np.zeros(len(books))
    total_w = 0
    for isbn in train_isbns:
        if isbn in isbn_to_idx:
            w = user_ratings_map.get(isbn, 1)
            cb_scores += content_sim[isbn_to_idx[isbn]].toarray().flatten() * w
            total_w += w
    if total_w > 0:
        cb_scores /= total_w

    hybrid = alpha * normalize(cb_scores) + (1 - alpha) * normalize(cf_scores)
    rated_idx = [isbn_to_idx[i] for i in train_isbns if i in isbn_to_idx]
    hybrid[rated_idx] = -np.inf
    top_idx = np.argsort(hybrid)[::-1][:n]
    return set(books.iloc[top_idx]["ISBN"])


# ---------- RMSE (rating prediction accuracy, on test set) ----------
preds, actuals = [], []
for _, r in test.iterrows():
    preds.append(predict_rating(r["User-ID"], r["ISBN"]))
    actuals.append(r["Book-Rating"])
rmse = np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))
print(f"RMSE (collaborative component): {rmse:.4f}")

# ---------- Precision@K / Recall@K (ranking quality) ----------
K = 10
RELEVANCE_THRESHOLD = 7  # a test rating >= 7 counts as "relevant"

test_relevant = test[test["Book-Rating"] >= RELEVANCE_THRESHOLD]
eval_users = test_relevant["User-ID"].unique()
# sample a subset of users for speed
eval_users = np.random.choice(eval_users, size=min(300, len(eval_users)), replace=False)

precisions, recalls = [], []
for uid in eval_users:
    train_isbns = train[train["User-ID"] == uid]["ISBN"].tolist()
    relevant_isbns = set(test_relevant[test_relevant["User-ID"] == uid]["ISBN"])
    if not relevant_isbns:
        continue
    recommended = hybrid_topn_for_user(uid, train_isbns, alpha=0.5, n=K)
    hits = recommended & relevant_isbns
    precisions.append(len(hits) / K)
    recalls.append(len(hits) / len(relevant_isbns))

print(f"Precision@{K}: {np.mean(precisions):.4f}")
print(f"Recall@{K}: {np.mean(recalls):.4f}")
print(f"Evaluated on {len(precisions)} users")
