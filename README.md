# Milling Tool Wear Detection

This project leverages deep learning and computer vision techniques to classify the wear state of milling tools. The goal is to provide an efficient solution for real-time monitoring and predictive maintenance in manufacturing, which can reduce downtime and ensure high-quality output.

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Use Cases](#use-cases)
- [About NJUST-CCTD](#about-njust-cctd)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Accuracies](#results-and-accuracies)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The project detects wear on milling tools by analyzing images of the tool surface using deep learning models. By classifying the wear level, we enable proactive maintenance, which is essential for high-volume and high-precision manufacturing environments. This classification is performed as:
1. **Two-Class Classification** - Identifying whether the tool has `No tool wear` or `Tool wear`.
2. **Three-Class Classification** - Categorizing wear level as `Severe tool wear`, `Mild tool wear`, or `No tool wear`.

The following deep learning models were used to achieve the results:
- **CNN (Convolutional Neural Network)**
- **VGG16** for feature extraction
- **Elastic Net** and **DSSNet** for improved regularization and model performance.

## Motivation

Milling tool wear detection is critical in manufacturing for maintaining tool health and ensuring product quality. As tools wear down over time, they produce parts with lower precision, affecting quality and causing potential delays due to unforeseen maintenance requirements. By monitoring the wear state, manufacturers can:
- Reduce costs associated with tool replacement and machine downtime.
- Improve product quality by ensuring that only well-conditioned tools are used in production.
- Enhance productivity with predictive maintenance strategies, scheduling tool changes only when necessary.

## Methodology

1. **Data Preprocessing**: The images are preprocessed to standardize their dimensions, enhance contrast, and apply noise reduction techniques to make wear features more discernible.
2. **Feature Extraction**: Using models like VGG16, important visual features related to tool wear are extracted, creating a robust set of inputs for classification.
3. **Model Training**: Several deep learning models are trained to classify tool wear. CNNs are particularly effective due to their ability to identify intricate wear patterns, while DSSNet and Elastic Net are used to provide additional regularization.
4. **Evaluation**: Models are evaluated based on accuracy, precision, and recall, with a focus on distinguishing wear levels in both two-class and three-class scenarios.
5. **Optimization**: Hyperparameter tuning and model selection are carried out to maximize accuracy, particularly for real-time application scenarios where precision is crucial.

## Models Used

1. **Convolutional Neural Network (CNN)**: Convolutional Neural Networks are used to detect features like cracks, chips, or wear lines in tool images, which can signify wear. CNNs are known for their effectiveness in image classification tasks and form the core of this project.
  
2. **VGG16**: This pre-trained model is used for feature extraction, providing robust features that enhance the CNN's ability to classify wear levels. VGG16 is beneficial in identifying minute details due to its depth and architecture.

3. **Elastic Net**: Elastic Net regularization is applied to reduce model complexity and prevent overfitting. By balancing L1 and L2 regularization, it stabilizes the learning process, especially in small to medium datasets like NJUST-CCTD.

4. **DSSNet**: DSSNet combines DenseNet and Squeeze-and-Excitation networks to achieve more nuanced feature selection, helping to increase accuracy by emphasizing wear regions in the images.

## Use Cases

1. **Predictive Maintenance**: Enabling timely maintenance by tracking wear levels to prevent sudden tool failure.
2. **Quality Assurance**: Maintaining consistent production quality by ensuring only tools in optimal condition are used.
3. **Cost Savings**: Reducing unnecessary tool replacements and minimizing downtime associated with unscheduled maintenance.
4. **Real-Time Monitoring**: Potential for integration into real-time monitoring systems to detect tool wear on the fly in manufacturing environments.

## About NJUST-CCTD

The **NJUST-CCTD** dataset is from the Nanjing University of Science and Technology, designed to support research in tool wear analysis. It contains high-quality images of cutting tools under varying wear conditions:
- **No Wear**
- **Mild Wear**
- **Severe Wear**

This dataset is critical for training models to recognize different wear levels, as it provides labeled images with clear distinctions in wear conditions. It captures images under multiple settings, making it suitable for real-world applications where lighting and tool positioning may vary.

## Dataset

The dataset used in this project consists of high-resolution images from the NJUST-CCTD, annotated according to wear levels:
- `No tool wear`
- `Mild tool wear`
- `Severe tool wear`

*Note: Usage of this dataset should comply with data usage policies.*

## Tool Wear Classification Project Flowchart

1. **Data Collection**  
   - Collect tool wear dataset (NJUST-CCTD).

2. **Data Preprocessing**  
   - **Image Resizing**: Resize images to a fixed dimension (e.g., 224x224 pixels).
   - **Normalization**: Scale pixel values to a range of 0 to 1.
   - **Noise Reduction**: Apply noise reduction filters to improve image clarity.
   - **Contrast Adjustment**: Adjust image contrast for better feature visibility.
   - **Data Augmentation**: Generate additional images by rotating, flipping, and zooming.

3. **Dataset Splitting**  
   - Split dataset into training, validation, and test sets.
   - Ensure class balance across each subset.

4. **Model Architecture Design**  
   - **Convolutional Layers**: Add layers with ReLU activation to extract features.
   - **Max-Pooling Layers**: Apply pooling to reduce data dimensions.
   - **Fully Connected Layers**: Integrate features for classification.
   - **Output Layer**: Use softmax activation for two-class probability output.

5. **Model Compilation**  
   - Choose optimizer (e.g., Adam) and loss function (categorical cross-entropy).
   - Set evaluation metric to accuracy.

6. **Model Training**  
   - Train model on training data, monitor validation performance.
   - Use early stopping and checkpointing to save the best model.

7. **Model Evaluation**  
   - Evaluate model on test set.
   - Generate accuracy, precision, recall, F1-score.
   - Visualize performance with confusion matrix and ROC curve.

8. **Model Fine-tuning**  
   - Adjust hyperparameters, retrain, and re-evaluate for improvement.

9. **Results Analysis**  
   - Analyze misclassifications to identify improvement areas.
   - Summarize final performance metrics and visualize key results.

## How to Use:
 1.  **Clone The Repository**

   ```bash
   git clone https://github.com/AnshSharma2521/Milling-Tool-Wear-Detection.git
   cd Milling-Tool-Wear-Detection
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Script**
    Run the jupyter codes

## Author
 **Ansh Sharma**





