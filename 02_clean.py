import pandas as pd
import re

DATA = "/home/claude/data"

books = pd.read_csv(f"{DATA}/BX_Books.csv", sep=";", encoding="latin-1",
                     on_bad_lines="skip", quotechar='"', low_memory=False)
users = pd.read_csv(f"{DATA}/BX-Users.csv", sep=";", encoding="latin-1",
                     on_bad_lines="skip", quotechar='"')
ratings = pd.read_csv(f"{DATA}/BX-Book-Ratings.csv", sep=";", encoding="latin-1",
                       on_bad_lines="skip", quotechar='"')

# --- Books ---
books = books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].copy()
books.dropna(subset=["Book-Title", "Book-Author", "Publisher"], inplace=True)
books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
books = books[(books["Year-Of-Publication"] > 1900) & (books["Year-Of-Publication"] <= 2026)]
books.drop_duplicates(subset="ISBN", inplace=True)
# Normalize text fields to avoid near-duplicate titles differing only by casing/whitespace
books["Book-Title"] = books["Book-Title"].str.strip()
books["Book-Author"] = books["Book-Author"].str.strip().str.title()
books.drop_duplicates(subset=["Book-Title", "Book-Author"], keep="first", inplace=True)

# --- Users ---
users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
users.loc[(users["Age"] < 5) | (users["Age"] > 100), "Age"] = None
users["Country"] = users["Location"].apply(
    lambda x: str(x).split(",")[-1].strip().lower() if pd.notna(x) else "unknown"
)
users.drop(columns=["Location"], inplace=True)

# --- Ratings ---
# Keep only explicit ratings (1-10); rating=0 means implicit interaction, not a preference score
ratings = ratings[ratings["Book-Rating"] > 0].copy()

# Keep only ratings referencing books/users we actually have
ratings = ratings[ratings["ISBN"].isin(books["ISBN"])]
ratings = ratings[ratings["User-ID"].isin(users["User-ID"])]

# Reduce sparsity: keep users with >=5 ratings and books with >=5 ratings (standard practice
# for Book-Crossing, since it's extremely sparse otherwise)
user_counts = ratings["User-ID"].value_counts()
book_counts = ratings["ISBN"].value_counts()
active_users = user_counts[user_counts >= 5].index
popular_books = book_counts[book_counts >= 5].index
ratings = ratings[ratings["User-ID"].isin(active_users) & ratings["ISBN"].isin(popular_books)]

# Keep only books/users that survived filtering
books = books[books["ISBN"].isin(ratings["ISBN"])]
users = users[users["User-ID"].isin(ratings["User-ID"])]

print("=== CLEANED SHAPES ===")
print("Books:", books.shape)
print("Users:", users.shape)
print("Ratings:", ratings.shape)
print("\nSparsity: %.4f%%" % (100 * (1 - len(ratings) / (books.shape[0] * users.shape[0]))))

books.to_csv(f"{DATA}/books_clean.csv", index=False)
users.to_csv(f"{DATA}/users_clean.csv", index=False)
ratings.to_csv(f"{DATA}/ratings_clean.csv", index=False)
print("\nSaved: books_clean.csv, users_clean.csv, ratings_clean.csv")
