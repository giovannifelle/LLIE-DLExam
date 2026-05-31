# LLIE-DLExam

Deep Learning exam project on **Low-Light Image Enhancement with Cross-Dataset
Generalization**.

The goal is to train a compact UNet on paired low-light images and evaluate how
well it generalizes to images from different datasets.

## Dataset Protocol

| Phase | Dataset | Usage |
| --- | --- | --- |
| Training | LOL-v2 Real Captured | Paired low-light and normal-light images |
| Validation | LOL-v2 Real Captured | Deterministic custom split from the training set |
| In-domain test | LOL-v2 Real Captured official test set | Paired evaluation |
| Cross-domain paired test | LOL-v1 `eval15` | Supervised evaluation |
| Cross-domain unpaired test | ExDark | NIQE, BRISQUE and qualitative analysis |

The datasets are not versioned in Git. They are expected under:

```text
data/raw/
    LOL-v2/
    lol_dataset/
    ExDark_dataset/
```

## Implemented Data Pipeline

Dataset loading is implemented with separate, explicit PyTorch dataset classes:

```text
src/
    datasets/
        base_dataset.py
        lolv1_dataset.py
        lolv2_dataset.py
        exdark_dataset.py
    preprocessing/
        split.py
        augmentation.py
        transforms.py
```

`LOLv2Dataset` and `LOLv1Dataset` return paired samples:

```python
{"low": low_image, "high": high_image}
```

`ExDarkDataset` returns unpaired samples with category metadata:

```python
{"image": image, "category": category, "path": image_path}
```

The preprocessing modules cover:

- deterministic dataset splitting with a fixed seed;
- TXT split files for LOL-v2 Real Captured and ExDark;
- RGB conversion, resize and tensor conversion;
- pixel normalization to the `[0, 1]` range;
- training-time random crop, horizontal flip and optional mild color jitter;
- deterministic validation and test transforms.

Preprocessing choices are controlled from `configs/config.yaml`.

## Generate Dataset Splits

Create the custom dataset splits with:

```bash
python3 scripts/make_splits.py
```

The generated files are:

```text
data/splits/
    lolv2_real_train.txt
    lolv2_real_val.txt
    ex_dark_split.txt
```

With the current seed and validation ratio, the split contains:

- `586` training pairs;
- `103` validation pairs.

The ExDark split contains `120` images: `10` deterministic samples for each of
its `12` categories. The TXT file stores relative paths such as
`Bicycle/2015_00001.png`.

The official LOL-v2 test set and LOL-v1 `eval15` are used entirely, so they do
not require custom split files.

## Current Dataset Sizes

| Dataset | Samples |
| --- | ---: |
| LOL-v2 Real Captured train split | 586 |
| LOL-v2 Real Captured validation split | 103 |
| LOL-v2 Real Captured official test set | 100 |
| LOL-v1 `eval15` | 15 |
| ExDark stratified evaluation split | 120 |

The current preprocessing target resolution is `256x256`.
