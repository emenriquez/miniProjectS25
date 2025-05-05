| Model                                                       | Accuracy (mean ± std)   |
|:------------------------------------------------------------|:------------------------|
| fullRun_SimpleCNN (5 epochs)                                | 85.72 ± 0.30            |
| fullRun_SimpleCNN (10 epochs)                               | 86.57 ± 0.20            |
| fullRun_SimpleCNN (15 epochs)                               | 86.50 ± 0.24            |
| fullRun_ImprovedCNN (15 epochs)                             | 88.32 ± 0.19            |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3)                | 88.22 ± 0.10            |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug     | 86.81 ± 0.09            |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler | 88.11 ± 0.26            |
| fullRun_MLPBaseline (15 epochs)                             | 83.84 ± 0.36            |
| fullRun_ImprovedCNN (15 epochs, Temp Scaling)               | 91.79 ± 2.41            |