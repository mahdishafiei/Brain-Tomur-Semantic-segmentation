# A quantitative comparison between Focal loss and Binary Cross-Entropy loss in Brain Tumor Auto-segmentation using U-net!
Brain tumors are among the fatal cancers and cause the death of many people annually. Early diagnosis of a brain tumor can help save the patient's life.

We have collected a dataset consisting of 314 brain MRI images in all planes taken by giving a contrast medium with the dimension of 800*512, which offers the highest resolution. First, skull stripping has been implemented to separate the brain from other parts in the images. Next, we have annotated the tumors in the images under the supervision of experienced radiologists to create ground truth. 

Comprehensive literature review on essential loss functions commonly used in image segmentation with U-net has been done, specifically focusing on Tversky Loss, Binary Cross-Entropy loss function Dice Loss, and Focal Loss. To determine the most effective model versions for all three loss functions, hyperparameter tuning was performed. The number of epochs and alpha values associated with the loss functions were optimized during this process. Following the comparison, the study further evaluates the effectiveness of two loss functions, Binary Cross-Entropy (BCE) and Focal loss, specifically in handling tumor regions within the dataset. Subsequently, the U-net convolutional neural network was implemented on the dataset using the two loss functions.


The two proposed loss functions were evaluated using 5-fold cross-validation, and the average precision, recall, and f1 were 76.16%, 71.9%, and 74.52 for BCE loss and 82.92%, 79.32%, and 81% for the Focal loss on the test data, respectively. Moreover, the accuracy for BCE loss was 99.03% and 99.44 % for the Focal loss.


It is evident that precision, recall, accuracy, and dice coefficient (f1 score) obtained from the U-net model with Focal loss are significantly higher than the U-net model with BCE loss. Based on the results, training U-net on an enriched database containing more than three hundred brain MRI images with high resolution makes it possible to locate and diagnose the tumor in the brain and achieve high Precision in the brain tumor semantic segmentation task.

