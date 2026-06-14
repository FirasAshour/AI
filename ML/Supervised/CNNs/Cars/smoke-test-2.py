import os
import pandas as pd
import config
import step1_data

# Build splits from step1_data
class_to_images, class_names = step1_data.scan_dataset(config.DATA_DIR)
train_split, val_split, test_split = step1_data.split_dataset(
    class_to_images,
    class_names,
    config.TRAIN_RATIO,
    config.VAL_RATIO,
    config.RANDOM_SEED,
)

# Wrap the (filepath, label_idx) tuples into DataFrames
cols = ["filepath", "label_idx"]
train_df = pd.DataFrame(train_split, columns=cols)
val_df   = pd.DataFrame(val_split,   columns=cols)
test_df  = pd.DataFrame(test_split,  columns=cols)

num_classes = len(class_names)

print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  classes={num_classes}")
print(train_df.head(3))