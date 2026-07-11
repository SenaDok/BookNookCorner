import pandas as pd

DATA = "/home/claude/data"

books = pd.read_csv(f"{DATA}/BX_Books.csv", sep=";", encoding="latin-1",
                     on_bad_lines="skip", quotechar='"', low_memory=False)
users = pd.read_csv(f"{DATA}/BX-Users.csv", sep=";", encoding="latin-1",
                     on_bad_lines="skip", quotechar='"')
ratings = pd.read_csv(f"{DATA}/BX-Book-Ratings.csv", sep=";", encoding="latin-1",
                       on_bad_lines="skip", quotechar='"')

print("=== SHAPES ===")
print("Books:", books.shape)
print("Users:", users.shape)
print("Ratings:", ratings.shape)

print("\n=== COLUMNS ===")
print("Books:", list(books.columns))
print("Users:", list(users.columns))
print("Ratings:", list(ratings.columns))

print("\n=== SAMPLE ===")
print(books.head(3))
print(users.head(3))
print(ratings.head(3))

print("\n=== RATING DISTRIBUTION ===")
print(ratings["Book-Rating"].describe())
print("\nImplicit (0) ratings count:", (ratings["Book-Rating"] == 0).sum())
print("Explicit (1-10) ratings count:", (ratings["Book-Rating"] != 0).sum())

print("\n=== MISSING VALUES ===")
print("Books:\n", books.isna().sum())
print("Users:\n", users.isna().sum())
