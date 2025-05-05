| Model                                                         | Accuracy (mean ± std)   |
|:--------------------------------------------------------------|:------------------------|
| BASE_RUN2_SimpleCNN (5 epochs)                                | 98.47 ± 0.17            |
| BASE_RUN2_SimpleCNN (10 epochs)                               | 98.58 ± 0.07            |
| BASE_RUN2_SimpleCNN (15 epochs)                               | 98.85 ± 0.11            |
| BASE_RUN2_ImprovedCNN (15 epochs)                             | 99.09 ± 0.11            |
| BASE_RUN2_ImprovedCNN (15 epochs, Dropout=0.3)                | 99.01 ± 0.13            |
| BASE_RUN2_ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug     | 98.75 ± 0.17            |
| BASE_RUN2_ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler | 99.08 ± 0.07            |
| BASE_RUN2_MLPBaseline (15 epochs)                             | 97.81 ± 0.21            |
| BASE_RUN2_ImprovedCNN (15 epochs, Temp Scaling)               | 99.65 ± 0.29            |