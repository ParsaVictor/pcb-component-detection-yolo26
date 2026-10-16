# Weights

| File | Model | Notes |
|---|---|---|
| `any_best.pt` | YOLO26s, 1 class `component` | locator, 1024 px tiles |
| `fine_best.pt` | YOLO26s, 13 classes | type detector, 1024 px tiles |
| `tree_rf.joblib` | Random Forest, 33 features | trained on isolated component photos (93% held-out) |
| `tree_rf_boards.joblib` | Random Forest, 33 features | trained on training-board crops (89% at p>=0.5) |
