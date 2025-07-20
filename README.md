# DEEP-LEARNING-PROJECT

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : VEDANT RAMESH KAWARE

*INTERN ID* : CT08DN595

*DOMAIN* : DATA SCIENCE

*DURATION* : 8 WEEKS

*MENTOR* : NEELA SANTOSH

In this project, we built a Convolutional Neural Network (CNN) to classify images of garbage into six meaningful categories: cardboard, glass, metal, paper, plastic, and trash. The goal was to apply deep learning, specifically using TensorFlow and Keras, to solve a real-world image classification task. The idea of automating waste classification holds tremendous value in promoting effective waste management systems, minimizing manual sorting, and improving the speed and accuracy of recycling operations. This work aligns with broader sustainability and environmental goals by offering a potential prototype for intelligent, AI-driven waste segregation systems.

The dataset for this project was sourced from Kaggle and contains approximately 2,500 labeled images distributed across six folders — each representing a different class of garbage. These folders served as class labels and were loaded using Keras’ flow_from_directory method. The images were preprocessed by resizing them to 128×128 pixels, rescaling their pixel values between 0 and 1, and splitting the data into training and validation sets in an 80:20 ratio. This preparation ensured that the data was in a suitable format for training a deep learning model efficiently and accurately.

The model architecture was built using TensorFlow's Keras Sequential API. It consisted of multiple convolutional layers to extract spatial features, followed by max-pooling layers to reduce dimensionality, and dropout layers to prevent overfitting. A dense layer with ReLU activation and a final softmax layer was used to classify the input image into one of the six garbage categories. We used the Adam optimizer for training and categorical cross-entropy as the loss function due to the multi-class nature of the problem. EarlyStopping was implemented to prevent overfitting by monitoring validation loss and stopping the training if no improvement was seen for 5 consecutive epochs. The model was set to train for 30 epochs, but early stopping halted the process at epoch 16, indicating good convergence.

To evaluate the model’s performance, we plotted training and validation accuracy and loss graphs. These visualizations allowed us to observe the model’s learning trends and determine whether it was overfitting or underfitting. We also made predictions on a random batch of validation images and compared the predicted labels to the true labels. This visual comparison helped validate how well the model was generalizing to unseen data. In addition, the project could optionally be extended with a confusion matrix and classification report for a more detailed performance analysis.

This project was implemented entirely using Python in a Jupyter Notebook. The main tools and libraries used included TensorFlow/Keras for model development, NumPy for numerical operations, Matplotlib for visualization, and scikit-learn for evaluation metrics. The entire project was organized inside a Project2 folder, with all resources, datasets, and code files neatly arranged for clarity and ease of understanding.

From a practical perspective, this project has significant real-world applications. With improvements such as larger and more balanced datasets, data augmentation, or the use of pre-trained models, this garbage classification system could be integrated into smart bins, automated waste-sorting lines, or robotic systems in industries and municipalities. It addresses a relevant environmental challenge and provides an example of how AI can be applied meaningfully for public good.

Overall, this project helped solidify my understanding of convolutional neural networks, image classification techniques, and TensorFlow-based model building. I learned how to handle and preprocess real-world image data, design an effective deep learning architecture, evaluate and visualize model performance, and apply good practices such as early stopping and dropout to improve generalization. It was a rewarding and insightful experience that combined technical learning with real-world impact.

Dataset Source: [Garbage Classification - Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

Saved Model(since size exceeds limit) - https://drive.google.com/file/d/1Q6usHRQe0j38zGjsnFWKe9xK-XbjKbrZ/view?usp=drive_link

<img width="1473" height="338" alt="Image" src="https://github.com/user-attachments/assets/02ec68cc-6fe3-4f59-bdb0-af5eab22c773" />
