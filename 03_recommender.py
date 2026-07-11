import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

DATA = "/home/claude/data"

books = pd.read_csv(f"{DATA}/books_clean.csv")
ratings = pd.read_csv(f"{DATA}/ratings_clean.csv")

books.reset_index(drop=True, inplace=True)
isbn_to_idx = {isbn: i for i, isbn in enumerate(books["ISBN"])}
idx_to_isbn = {i: isbn for isbn, i in isbn_to_idx.items()}

# ---------- CONTENT-BASED FILTERING ----------
# Feature text: title + author + publisher (no genre field exists in Book-Crossing)
books["content"] = (
    books["Book-Title"].fillna("") + " " +
    books["Book-Author"].fillna("") + " " +
    books["Publisher"].fillna("")
)
tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(books["content"])
content_sim = cosine_similarity(tfidf_matrix, dense_output=False)  # sparse-friendly


def content_based_scores(user_isbns, user_ratings_map):
    """Weighted avg similarity of all candidate books to the user's rated books."""
    scores = np.zeros(len(books))
    total_weight = 0
    for isbn in user_isbns:
        if isbn not in isbn_to_idx:
            continue
        idx = isbn_to_idx[isbn]
        weight = user_ratings_map[isbn]  # use rating (1-10) as confidence weight
        sims = content_sim[idx].toarray().flatten()
        scores += sims * weight
        total_weight += weight
    if total_weight > 0:
        scores /= total_weight
    return scores


# ---------- COLLABORATIVE FILTERING (matrix factorization via Truncated SVD) ----------
user_ids = ratings["User-ID"].unique()
user_id_to_idx = {u: i for i, u in enumerate(user_ids)}

row = ratings["User-ID"].map(user_id_to_idx)
col = ratings["ISBN"].map(isbn_to_idx)
data = ratings["Book-Rating"].values

user_item_matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(books)))

N_COMPONENTS = 30
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)   # (n_users, k)
item_factors = svd.components_.T                      # (n_items, k)


def collaborative_scores(user_id):
    if user_id not in user_id_to_idx:
        return np.zeros(len(books))
    u_idx = user_id_to_idx[user_id]
    return item_factors @ user_factors[u_idx]


# ---------- HYBRID MERGE ----------
def normalize(scores):
    rng = scores.max() - scores.min()
    return (scores - scores.min()) / rng if rng > 0 else scores


def hybrid_recommend(user_id, top_n=10, alpha=0.5):
    """alpha: weight for content-based; (1-alpha) for collaborative."""
    user_ratings = ratings[ratings["User-ID"] == user_id]
    user_isbns = user_ratings["ISBN"].tolist()
    user_ratings_map = dict(zip(user_ratings["ISBN"], user_ratings["Book-Rating"]))

    cb_scores = normalize(content_based_scores(user_isbns, user_ratings_map))
    cf_scores = normalize(collaborative_scores(user_id))

    hybrid_scores = alpha * cb_scores + (1 - alpha) * cf_scores

    # exclude already-rated books
    already_rated_idx = [isbn_to_idx[i] for i in user_isbns if i in isbn_to_idx]
    hybrid_scores[already_rated_idx] = -np.inf

    top_idx = np.argsort(hybrid_scores)[::-1][:top_n]
    recs = books.iloc[top_idx][["ISBN", "Book-Title", "Book-Author"]].copy()
    recs["Score"] = hybrid_scores[top_idx]
    return recs.reset_index(drop=True)


if __name__ == "__main__":
    sample_user = ratings["User-ID"].value_counts().index[0]  # most active user
    print(f"Sample User ID: {sample_user}")
    print(f"Number of books rated: {(ratings['User-ID'] == sample_user).sum()}")
    print("\nTop 10 Hybrid Recommendations:\n")
    print(hybrid_recommend(sample_user, top_n=10, alpha=0.5).to_string(index=False))
