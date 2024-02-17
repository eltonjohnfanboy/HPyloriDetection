# H. pylori Detection

This repository presents two distinct methodologies for the detection of Helicobacter pylori (H. pylori) infections within histological images with immunohistochemical staining, leveraging two different deep learning methods. The implemented techniques include **autoencoders** and **contrastive learning**.

## Methodologies

### Autoencoder Approach

![image](https://github.com/eltonjohnfanboy/HPyloriDetection/assets/103358618/bbd6773d-2043-4e16-9393-31ed6e510b48)

The autoencoder method involves training the model with negative images (images without the infection). During the training phase, the autoencoder learns to reconstruct the negative images accurately, but, when confronted with positive images containing H. pylori, the autoencoder encounters difficulty in accurately reconstructing the pertinent regions due to its exclusive exposure to negative examples. 

![image](https://github.com/eltonjohnfanboy/HPyloriDetection/assets/103358618/adf9e5d1-4296-46fa-879d-579b9908ad74)

By establishing a threshold for image features indicative of H. pylori presence, we can effectively classify images into positive and negative categories.

![image](https://github.com/eltonjohnfanboy/HPyloriDetection/assets/103358618/6a1828d2-dc31-4700-910d-68f4dca6d5dc)

### Contrastive Learning Approach

The second methodology employs contrastive learning, employing the triplet loss method. 
![image](https://github.com/eltonjohnfanboy/HPyloriDetection/assets/103358618/2d09ca13-b8cf-4154-b4c5-b91a90c6a1d1)

Within this framework, the model is trained using triplets of samples, comprising an anchor image, a positive image, and a negative image. The model endeavors to discern between positive and negative samples by optimizing the distances between the anchor image and both positive and negative counterparts. Through this iterative training process, the model acquires the capability to distinguish features characteristic of H. pylori presence.
![image](https://github.com/eltonjohnfanboy/HPyloriDetection/assets/103358618/a8f1cd9c-9708-48d1-a859-ed5b9c9e6687)
As observed in the previous figures, the model converges successfully and it's able to cluster correctly the two classes (infected and not infected).


## Repository Structure

The code for implementing the two approaches resides in the following folders:
- **Autoencoder**: Contains code related to the autoencoder approach.
- **ContrastiveWithTripletLoss**: Contains code related to the contrastive learning approach using triplet loss.

Feel free to explore the respective folders for more details on the implementation of each method.

## Contributing

We welcome contributions to enhance and improve this H. pylori detection project. If you have any suggestions, ideas, or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
