If you get FileNotFoundError when loading the CSV in the notebook:

- If your notebook code uses: pd.read_csv("Amazon_Reviews.csv")
  change it to: pd.read_csv("../data/Amazon_Reviews.csv")

Or use a robust path snippet:
from pathlib import Path
DATA_PATH = Path.cwd() / "data" / "Amazon_Reviews.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path.cwd().parent / "data" / "Amazon_Reviews.csv"
df = pd.read_csv(DATA_PATH)
