from types import SimpleNamespace

conf = SimpleNamespace(
    root_dir = "data/brain-tumor-mri-dataset/",
    work_dir = "data/work_dir/",
    train_dir = "data/work_dir/train/",
    test_dir = "data/work_dir/test/",
    val_dir = "data/work_dir/val/",
)

train_conf = SimpleNamespace(
    img_shape = (384, 384),
    batch_size = 16,
    epochs = 5
)

test_conf = SimpleNamespace(
    latest_best = "models/brain_mri_4_advanced_catg-23 may.keras"
)