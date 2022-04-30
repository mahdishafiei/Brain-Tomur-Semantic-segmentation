# Brain-Tomur-Semantic-seg
Collaboration on advancing science (Seriously!)
here we are going to share the code of this project 


1.	Introduction
Brain tumors are among the most lethal cancers and cause the death of a significant number of people annually. An estimated 16,830 deaths were attributed to primary malignant brain tumors in the US in 2018. According to a study from 2011 to 2015, In the US, the 5-year and 10-year relative survival rates of patients [1] with malignant tumors were 35.0% and 29.3%, respectively. Based on this study, Brain and other Central nervous system tumors (both malignant and non-malignant) had an average annual age-adjusted incidence of 11.20 per 100,000 people aged 15–39 years. Also, these tumors were the second most common cancer in males in this age group and the third most common cancer in females in this age group [1].

Magnetic resonance imaging (MRI) is the primary scan tool for detecting and diagnosing tumors since MRI provides outstanding contrast for soft tissues. In general, the function of MR imaging in the workup of Intra axial tumors can be divided into tumor diagnosis and classification, treatment planning, and post-treatment management [2]. A particular dye called a contrast medium is given before the scan to make a more precise image.

Image segmentation is the process of partitioning a digital image into some segments. Image Segmentation is commonly used to find objects and edges in the image [3]. Also, image segmentation is the process of allocating a label to every pixel in an image in which the pixels with the same label share specific visual characteristics [4].

Brain tumor image segmentation is often done by experts to separate different parts of the brain image to identify tumors and differentiate between them and normal parts of the brain. In the medical routine, Precise segmentation of brain tumors provides valuable information for treatment planning. Process of tumor localization is very intense work and highly dependent on the doctors’ experience, skills, and their slice-by-slice decisions. Early diagnosis of a cancerous tumor can increase the chance of the patient's recovery after treatment and improve the survival probability of the patient. However, qualified experts need a considerable period of time to segment the brain tumors. In addition, this process is prone to be influenced by individual opinions, and the results may vary between observers. 

Many computers aided diagnosis methods have been developed in recent years to automate the process of segmentation to save the doctors' time and to provide trustworthy and accurate results while reducing the exerted efforts of experienced physicians to perform the procedures of diagnosis for every single patient. Some of the special segmentation models are:

•	Thresholding method: if the value for each pixel exceeds the threshold, it is identified as a tumor. [5].
•	Edge-based method: differences in the intensity between edges of pixels are used as the boundaries of the tumors [6].
•	Region growing method: first, a pixel input into segmentation and similar pixels will be classified as a tumor. [7].
•	Atlas method: An MRI without a tumor will be used to segment MRI with the tumor volume. [8].

Among these methods, the role of Convolutional Neural Networks (CNN) in automated brain tumor segmentation is meaningful and attracted great interest in recent years. CNN was introduced in 1990, inspired by experiments performed by Hubel and Wiesel. One of the first projects to be done with CNN was Yann Lecun's famous handwritten digit identification project. He proposed the LeNet-5, which produced promising results [9]. The central brilliance of the convolution network occurred in 2012 during the ImageNet Large Scale Recognition Challenge. This was an event that caught the attention of the research community. In the 2012 ImageNet Contest, Alex Krizhevsky entered with the AlexNet, which had an excellent performance compared to others in classifying images with 16.4% error. 

One of the most helpful CNN architectures is U-Net [10], which introduces skip connections between the layers in the network. The U-net architecture achieves excellent performance on various biomedical segmentation applications. However, Isensee et al. [11], who acquired top performance using a well-trained U-Net, showed that enhancing segmentation performance is not just a matter of adjusting the network architecture. The choice of the loss function, training strategy, and post-processing led had an enormous impact on the segmentation performance.

Semantic segmentation Is a cardinal task in computer vision, and it aims to map each pixel in an image to its associated label, like tumor, or normal tissue.. Some of the earliest approaches to implementing this method in the medical field suffer from two main problems:
First, the training patches (local regions) are larger than the training samples. As a result, the running time will be significantly prolonged. Second, the accuracy of the segmentation depends on the appropriate size of the patches. Consequently, the U-net architecture has been introduced to conquer these problems [10].


As is evident in Figure (1), U-net includes two main paths: a contracting path (or encoder) and an expansive path (or decoder)[10]. The encoder is a CNN consisting of two consecutive 3×3 convolutional (unpadded ) layers, separately followed by a rectified linear unit (ReLU) and a 2×2 max-pooling layer. Conversely, the decoder seeks to upsample the resulting feature map using deconvolution layers followed by 2×2 up-convolution, a concatenation layer with the corresponding downsampled layer from the encoder, two 3×3 convolutions, and a ReLU [12].
Finally, the upsampled features will be conducted to a 1× 1 convolution layer to output the final segmentation map. As a result of this process, the networks can achieve precise segmentation outcomes. 


