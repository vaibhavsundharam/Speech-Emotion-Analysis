# Introduction
Human emotions are one of the strongest ways of communication.  Even if a person doesn’t understand a language, he or she can very well understand the emotions delivered by an individual.  In other words, emotions are universal.The idea behind the project is to develop a Speech Emotion Analyzer using deep-learning to correctly classify a human’s different emotions, such as, neutral speech, angry speech, surprised speech, etc. We have deployed three different network architectures namely 1-D CNN, LSTMs and Transformers to carryout the classification task. Also, we have used two different feature extraction methodologies (MFCC &amp; Mel Spectrograms) to capture the features in a given voice signal and compared the two in their ability to produce high quality results, especially in deep-learning models.

# Data
There are many datasets available for audio emotion recognition. Many of them consist of trained actors who enunciate short phrases in various emotions. Some commonly used datasets for emotion classification are SAVEE (Surrey Audio-Visual Expressed Emotion) [14] and TESS (Toronto Emotional Speech) [15]. SAVEE dataset consists of only four male speakers whereas, the TESS dataset consists of only two female speakers. It is important to note that both male and female voices are required to design a novel emotion classification neural network as we do not want the network to be biased towards only one gender. The issue with these two datasets is that even though they have good quality audio files, both have very limited in terms of variety of voice actors. We wanted a dataset that had a larger number of voice actors as compared to the aforementioned datasets. Hence, we decided to use the RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song) [16] dataset. RAVDESS is one of the most frequently used datasets for this task as it consists of 12 female and 12 male speakers who are showcasing eight different emotions, namely, neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Each speaker is enunciating two sentences ("Kids are talking by the door" and "Dogs are sitting by the door") in eight different emotions and with two different intensities (strong and normal) except for the neutral emotion, which only has a normal intensity. Hence, each actor has 60 enunciations making a total of 1,440 audio files.

Datsets:  [SAVEE Dataset](https://www.kaggle.com/barelydedicated/savee-database), [TESS Dataset](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess), [RAVDESS Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

# Data pre-processing
The classification network should be trained separately on male and female voices. This is because males and females have very different physiologic and acoustic features [17, 18]. For example, a female voice has a higher pitch as compared to a male as depicted in Figure 1. Hence, we separated male and female voices in the dataset. Each subset is then split into training, validation, and testing set in the ratio of 8:1:1.
![Alt text]https://github.com/vaibhavsundharam/Speech-Emotion-Analysis/blob/main/Images/Figure%201.png?raw=true)

