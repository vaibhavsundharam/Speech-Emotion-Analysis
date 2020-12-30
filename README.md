# Introduction
Human emotions are one of the strongest ways of communication.  Even if a person doesn’t understand a language, he or she can very well understand the emotions delivered by an individual.  In other words, emotions are universal.The idea behind the project is to develop a Speech Emotion Analyzer using deep-learning to correctly classify a human’s different emotions, such as, neutral speech, angry speech, surprised speech, etc. We have deployed three different network architectures namely 1-D CNN, LSTMs and Transformers to carryout the classification task. Also, we have used two different feature extraction methodologies (MFCC &amp; Mel Spectrograms) to capture the features in a given voice signal and compared the two in their ability to produce high quality results, especially in deep-learning models.

# Data
There are many datasets available for audio emotion recognition. Many of them consist of trained actors who enunciate short phrases in various emotions. Some commonly used datasets for emotion classification are SAVEE (Surrey Audio-Visual Expressed Emotion) [14] and TESS (Toronto Emotional Speech) [15]. SAVEE dataset consists of only four male speakers whereas, the TESS dataset consists of only two female speakers. It is important to note that both male and female voices are required to design a novel emotion classification neural network as we do not want the network to be biased towards only one gender. The issue with these two datasets is that even though they have good quality audio files, both have very limited in terms of variety of voice actors. We wanted a dataset that had a larger number of voice actors as compared to the aforementioned datasets. Hence, we decided to use the RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song) [16] dataset. RAVDESS is one of the most frequently used datasets for this task as it consists of 12 female and 12 male speakers who are showcasing eight different emotions, namely, neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Each speaker is enunciating two sentences ("Kids are talking by the door" and "Dogs are sitting by the door") in eight different emotions and with two different intensities (strong and normal) except for the neutral emotion, which only has a normal intensity. Hence, each actor has 60 enunciations making a total of 1,440 audio files.

Datsets:  [SAVEE Dataset](https://www.kaggle.com/barelydedicated/savee-database), [TESS Dataset](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess), [RAVDESS Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

# Data pre-processing
The classification network should be trained separately on male and female voices. This is because males and females have very different physiologic and acoustic features [17, 18]. For example, a female voice has a higher pitch as compared to a male as depicted in Figure 1. Hence, we separated male and female voices in the dataset. Each subset is then split into training, validation, and testing set in the ratio of 8:1:1.

![Figure 1](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure_1.png?raw=true)
Figure (1) Difference between the pitch of a male and a female voice actor enunciating the same dialog with the same emotion (angry)

# Feature Extraction
The first approach for extracting features from the voice signal was based on Fourier transforms. Unfortunately, this approach was flawed as Fourier transforms are not useful in showcasing how a human individual perceives sound. Hence, we decided to use Mel Frequency Cepstral Coefficients (MFCC) [19] and Mel Spectrograms [20]. Generally, we take the Fourier transform to decompose a time-domain signal into its frequency components. But, for calculating MFCC, the cosine transformation of the log of the magnitude of the Fourier spectrum is taken. Hence, the signal is neither in the time domain nor in the frequency domain, rather, it is in the quefrency domain [21] with its spectrum
termed as cepstrum. Cepstrum gives us information about the rate of change of spectral bands. The vocal tract (including tongue, teeth, etc.) determines the sound generated and, the MFCC accurately represents the envelope of the time power spectrum of a speech signal which is a representation of the vocal tract. Figure 2a and Figure 2b represent the MFCC plots of a male and a female speaker respectively showcasing angry emotion. We can see that both plots look different hence affirming that males and females have different acoustic features [17, 18].

![Eq 1](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Eq_1.png?raw=true)


![Figure 2](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure_2.png?raw=true)
Figure (2) MFCC plots of a male and a female speaker showcasing angry emotion

Mel spectrograms are calculated by taking the log of the frequency components obtained after the FFT of an audio signal and converting the amplitude of the audio signal to decibels. The frequency is mapped to the Mel scale using Eq. 1 [21]. Figure 3 shows the Mel spectrograms for a male and female audio showcasing the angry emotion. The difference between the acoustic features of a male and a female can also be seen from the aforementioned figure.

![Figure 3](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure_2.png?raw=true)
Figure (3) Spectrograms of a male and female speaker showcasing ’angry’ emotion

# Data Augmentation
Training an artificial neural network usually requires a sufficiently large amount of data. To circumvent this issue, various augmentations can be applied to synthetically generate new data points. For example, for audio data, augmentations include adding noise, shifting the audio, changing the pitch, stretching the audio, etc. In our implementation, we have added white Gaussian noise to the voice signals. We have synthesized two noise signals sampled from a normal distribution and added them
to the voice signal thereby, increasing the number of training examples from 576 to 1728 for both male and female data subsets. This augmentation not only increased the number of data points but also helped in making the model much more robust.

# Model Architectures
We implemented three different neural network architectures to perform the classification task. The first architecture is the baseline model and is built using 1-D convolution neural networks. The second and the third model that we implemented use Transformers and Long Short Term Memory (LSTM) architectures at their core. In the following section, we will deep dive into each of the network architectures.

## Baseline Model 
Convolution neural networks (CNNs) have shown their prowess in various machine learning applications ranging from computer vision tasks to natural language processing (NLP). In today’s world, there is no domain left untouched by CNNs. Hence, our first obvious choice was to use 1-D CNNs to build the baseline model. Figure 4 depicts the block diagram of the baseline model we implemented. Each Conv 1D Block consists of a conv1d layer followed by batch normalization and relu activation. Dropout layers were introduced at various locations throughout the model to circumvent the issue of overfitting.

![Figure 4](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure_4.png?raw=true)
Figure (4) Block diagram of the baseline model

## LSTM Model
Recurrent neural networks (RNNs) were one of the first successful models deployed for natural language processing applications as they allow information to persist. This is useful, especially for sequential data like speech and language, where the prediction of the next word in the sentence/speech depends upon the phrases that came before. At each level inside an RNN, information from the previous layers gets transferred to the next layer. This information transfer creates a dependence between subsequent layers inside the network. But, RNNs fail to learn long-term dependencies, i.e. they perform poorly when the gap between relevant information and the place where it is required
becomes large. Long-Short Term Memory (LSTM) [25, 26] networks with attention mechanisms were introduced to address some of the drawbacks of RNNs. LSTMs are also sequential models. But, unlike RNNs, they can selectively store and discard information that is important and not so important. In our implementation, we have used a network architecture inspired by bidirectional LSTMs** to perform the classification task. The network architecture of the model is depicted in Figure 5.
![Figure 5](https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure_5.png?raw=true)
[**](https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch)
