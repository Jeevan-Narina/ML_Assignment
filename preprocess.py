import os
import numpy as np
import librosa

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = "Data/genres_original"
OUTPUT_PATH = "Data/processed"

GENRES = ["classical", "jazz", "metal", "pop"]

INPUT_DIM = 500
NUM_SPLITS = 20

# Original implementation:
# 600000 raw samples -> reshape into (15000, 40)
# -> average pooling -> 15000 pooled values
POOL_SIZE = 40
NUM_RAW_SAMPLES = 600000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# STORAGE
# ============================================================

X = []
y = []


# ============================================================
# PROCESS EACH GENRE
# ============================================================

for label, genre in enumerate(GENRES):

    genre_path = os.path.join(BASE_PATH, genre)

    print(f"\nProcessing genre: {genre}")

    files = sorted(
        [
            f
            for f in os.listdir(genre_path)
            if f.lower().endswith(".wav")
        ]
    )

    song_count = 0

    for filename in files:

        filepath = os.path.join(genre_path, filename)

        try:
            # LibROSA default sample rate = 22050 Hz
            audio, sr = librosa.load(filepath)

            # ------------------------------------------------
            # Original preprocessing
            # ------------------------------------------------

            # Take first 600000 raw audio samples
            audio = audio[:NUM_RAW_SAMPLES]

            # We need exactly 600000 samples
            if len(audio) < NUM_RAW_SAMPLES:
                print(f"Skipping {filename}: too short")
                continue

            # Reshape:
            # 600000 / 40 = 15000
            audio = audio.reshape(15000, POOL_SIZE)

            # Average pooling
            audio = np.mean(audio, axis=1)

            # ------------------------------------------------
            # Create 20 examples from each song
            # Each example has 500 values
            # ------------------------------------------------

            for j in range(NUM_SPLITS):

                start = j * INPUT_DIM
                end = (j + 1) * INPUT_DIM

                segment = audio[start:end]

                if len(segment) == INPUT_DIM:
                    X.append(segment)
                    y.append(label)

            song_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Successfully processed {song_count} songs")
    print(f"Generated {song_count * NUM_SPLITS} examples")


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)


# ============================================================
# SAVE DATA
# ============================================================

X_path = os.path.join(OUTPUT_PATH, "X_v4.npy")
y_path = os.path.join(OUTPUT_PATH, "y_v4.npy")

np.save(X_path, X)
np.save(y_path, y)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nClass distribution:")

for label, genre in enumerate(GENRES):
    count = np.sum(y == label)
    print(f"{genre}: {count}")

print("\nSaved:")
print(X_path)
print(y_path)