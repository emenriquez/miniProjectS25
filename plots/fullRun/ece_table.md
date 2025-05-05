| Model                                                       | ECE (mean ± std)   |
|:------------------------------------------------------------|:-------------------|
| fullRun_SimpleCNN (5 epochs)                                | 0.0085 ± 0.0034    |
| fullRun_SimpleCNN (10 epochs)                               | 0.0181 ± 0.0041    |
| fullRun_SimpleCNN (15 epochs)                               | 0.0329 ± 0.0044    |
| fullRun_ImprovedCNN (15 epochs)                             | 0.0115 ± 0.0026    |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3)                | 0.0258 ± 0.0035    |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3) + Data Aug     | 0.0059 ± 0.0015    |
| fullRun_ImprovedCNN (15 epochs, Dropout=0.3) + LR Scheduler | 0.0250 ± 0.0045    |
| fullRun_MLPBaseline (15 epochs)                             | 0.0455 ± 0.0037    |
| fullRun_ImprovedCNN (15 epochs, Temp Scaling)               | 0.0127 ± 0.0053    |