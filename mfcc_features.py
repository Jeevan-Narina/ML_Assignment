import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "Data/genres_original"
PROCESSED_DIR = "Data/processed"

GENRES = ["classical", "jazz", "metal", "pop"]

SR = 22050

# Same preprocessing idea as the original DSAE:
# 40 raw samples -> 1 pooled sample
POOL_SIZE = 40

# Original DSAE input:
# 500 pooled samples -> 20,000 raw samples
SEGMENT_LENGTH = 500 * POOL_SIZE

NUM_SEGMENTS = 20

# MFCC configuration
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# EXTRACT MFCC FEATURES FROM ONE AUDIO SEGMENT
# ============================================================

def extract_mfcc_features(segment, sr):
    """
    Extract MFCC + delta + delta-delta features.

    For each of the 13 MFCC coefficients:
        - mean
        - standard deviation

    We do this for:
        - MFCC
        - Delta MFCC
        - Delta-Delta MFCC

    Total:
        13 * 2 * 3 = 78 features
    """

    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),

            np.mean(delta, axis=1),
            np.std(delta, axis=1),

            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ]
    )

    return features


# ============================================================
# LOAD DATA AND EXTRACT MFCC FEATURES
# ============================================================

def extract_all_features():

    all_features = []
    all_labels = []

    print("=" * 60)
    print("MFCC FEATURE EXTRACTION")
    print("=" * 60)

    for label, genre in enumerate(GENRES):

        genre_dir = os.path.join(DATA_DIR, genre)

        if not os.path.exists(genre_dir):
            raise FileNotFoundError(
                f"Genre folder not found: {genre_dir}"
            )

        files = os.listdir(genre_dir)

        wav_files = [
            f for f in files
            if f.lower().endswith(".wav")
        ]

        print(f"\n{genre}: {len(wav_files)} audio files")

        for file_idx, filename in enumerate(wav_files):

            filepath = os.path.join(
                genre_dir,
                filename
            )

            try:

                # Same LibROSA loading convention as the
                # original preprocessing.
                audio, sr = librosa.load(
                    filepath,
                    sr=SR,
                    mono=True
                )

                # Same effective usable portion as
                # the original DSAE preprocessing.
                audio = audio[:600000]

                # Generate the same 20 temporal segments
                # used by the DSAE.
                for segment_idx in range(NUM_SEGMENTS):

                    start = segment_idx * SEGMENT_LENGTH
                    end = start + SEGMENT_LENGTH

                    segment = audio[start:end]

                    # Make sure the segment is complete.
                    if len(segment) < SEGMENT_LENGTH:
                        continue

                    mfcc_features = extract_mfcc_features(
                        segment,
                        sr
                    )

                    all_features.append(mfcc_features)
                    all_labels.append(label)

            except Exception as e:

                print(
                    f"ERROR processing {filename}: {e}"
                )

        print(
            f"Processed {genre}: "
            f"{len(all_features)} total examples so far"
        )

    X = np.asarray(
        all_features,
        dtype=np.float32
    )

    y = np.asarray(
        all_labels,
        dtype=np.int64
    )

    return X, y


# ============================================================
# VERIFY THAT LABEL ORDER MATCHES EXISTING DATASET
# ============================================================

def verify_split_labels(y_full):

    print("\n" + "=" * 60)
    print("VERIFYING DATA SPLIT")
    print("=" * 60)

    # Reproduce the same split strategy used for
    # the current project:
    #
    # 75% train
    # 25% temporary
    # temporary -> 50% dev + 50% test
    #
    # random_state = 42

    indices = np.arange(len(y_full))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=42,
        stratify=y_full
    )

    dev_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=y_full[temp_idx]
    )

    # Load current labels
    y_train_existing = np.load(
        os.path.join(PROCESSED_DIR, "y_train.npy")
    )

    y_dev_existing = np.load(
        os.path.join(PROCESSED_DIR, "y_dev.npy")
    )

    y_test_existing = np.load(
        os.path.join(PROCESSED_DIR, "y_test.npy")
    )

    train_match = np.array_equal(
        y_full[train_idx],
        y_train_existing
    )

    dev_match = np.array_equal(
        y_full[dev_idx],
        y_dev_existing
    )

    test_match = np.array_equal(
        y_full[test_idx],
        y_test_existing
    )

    print(f"Train labels match: {train_match}")
    print(f"Dev labels match:   {dev_match}")
    print(f"Test labels match:  {test_match}")

    if not (train_match and dev_match and test_match):

        print("\nWARNING:")
        print(
            "The generated MFCC labels do not match "
            "the existing project split."
        )

        raise ValueError(
            "Split verification failed. "
            "Do NOT use these MFCC features yet."
        )

    print("\nSplit verification successful.")

    return train_idx, dev_idx, test_idx


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    X, y = extract_all_features()

    print("\n" + "=" * 60)
    print("FULL DATASET")
    print("=" * 60)

    print("MFCC feature shape:", X.shape)
    print("Label shape:", y.shape)

    print("\nClass counts:")

    for label, genre in enumerate(GENRES):

        count = np.sum(y == label)

        print(
            f"{genre}: {count}"
        )

    # --------------------------------------------------------
    # Verify split
    # --------------------------------------------------------

    train_idx, dev_idx, test_idx = verify_split_labels(y)

    # --------------------------------------------------------
    # Create train/dev/test MFCC arrays
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_dev = X[dev_idx]
    X_test = X[test_idx]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = os.path.join(
        PROCESSED_DIR,
        "mfcc_train.npy"
    )

    dev_path = os.path.join(
        PROCESSED_DIR,
        "mfcc_dev.npy"
    )

    test_path = os.path.join(
        PROCESSED_DIR,
        "mfcc_test.npy"
    )

    np.save(train_path, X_train)
    np.save(dev_path, X_dev)
    np.save(test_path, X_test)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MFCC FEATURES SAVED")
    print("=" * 60)

    print(
        "MFCC train:",
        X_train.shape
    )

    print(
        "MFCC dev:",
        X_dev.shape
    )

    print(
        "MFCC test:",
        X_test.shape
    )

    print("\nFiles created:")

    print(train_path)
    print(dev_path)
    print(test_path)

    print("\nFeature dimension:", X_train.shape[1])

    print("\nDone.")


if __name__ == "__main__":
    main()