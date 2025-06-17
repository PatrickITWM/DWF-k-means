from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import resample

# Load the data
df = pd.read_parquet("raw.parquet") # Download data first manually from here https://huggingface.co/datasets/flwrlabs/femnist
print(f"Loaded the data. Shape: {df.shape}")
# Preprocess the images (byte content to np.array [1D])
print("Preprocessing the images...")
df["image"] = df["image.bytes"].apply(lambda x: np.array(Image.open(BytesIO(x))).reshape(28 * 28))
print("Preprocessing done. Now saving the data to disk. This may take a while.")

# Create X and y
X = np.stack(df["image"].to_numpy())  # Convert X to a single table
X = 255 - X  # Invert
X = X / 255 # Scale
y = df["character"].to_numpy()

# Resample to make it smaller
print("Resampling the data to make it smaller.")
X, y = resample(X, y, random_state=1024, replace=False, stratify=y, n_samples=100_000)

# Save it
print("Change data format")
X_df = pd.DataFrame(X)
X_df.columns = X_df.columns.astype(str)
y_df = pd.DataFrame(y)
y_df.columns = y_df.columns.astype(str)

print("Actually save the data to disk.")
X_df.to_parquet("X.parquet", compression="gzip")
y_df.to_parquet("y.parquet", compression="gzip")

print("Done.")
