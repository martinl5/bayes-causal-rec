# Data

The contents of this directory are gitignored. This file documents what goes
here and how to obtain it.

## Coat Shopping (primary, real MNAR benchmark)

- **What:** 290 users x 300 items, integer ratings on a 1-5 scale. The training
  split is biased (users self-selected which coats to rate); the test split is
  collected under uniform-random exposure and is therefore unbiased — exactly
  the MNAR structure this project corrects for.
- **Source:** https://www.cs.cornell.edu/~schnabts/mnar/ (Schnabel et al. 2016).
- **Layout after download:**
  ```
  data/raw/coat/train.ascii   # 290 x 300 dense integer matrix, 0 = unobserved
  data/raw/coat/test.ascii    # 290 x 300 dense integer matrix, 0 = unobserved
  ```
- **How:** `make data` (or `bcr-download --data-dir data/raw`). If the mirrors
  are unreachable, download the two `.ascii` files manually and place them in
  `data/raw/coat/`.

## Synthetic MNAR (always available fallback)

Generated in-process by `bcr.data.preprocess.make_synthetic_mnar`; nothing is
written to disk. Defaults: 500 users, 200 items, 10 latent factors, ratings in
[1, 5]. The exposure model is

```
logit P(observe | u, i) = alpha_popularity * log(item_popularity_i)
                          + alpha_relevance  * (true_rating_ui - mean)
```

so popular and well-liked items are over-represented in the training sample.
The generator returns the **known ground-truth propensities**, which is why the
reported calibration results use it: ECE can be checked against the true
exposure probabilities. All model code runs identically on Coat when present.
