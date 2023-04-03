# Motor-Imagery-BCI-Competition-IV-IIa
2-class classification of “BNCI 2014-001 Left-Right Motor Imagery” dataset.

This repo contains a source code analysis of left-right motor-imagery EEG data. It employs the Tangent Space method as feature extraction technique and then performs 2-class classification of the data examining 3 classifiers:
1. LDA
2. SVM 
3. kNN

Each model is intra-subject and the evaluation is performed accross the 2 different sessions of each subject. 


## Dataset Information
The dataset and its description can be found in:
http://moabb.neurotechx.com/docs/datasets.html

### Dataset Description
BNCI 2014-004 Motor Imagery dataset.
This data set consists of EEG data from 9 subjects. The cue-based BCI paradigm consisted of four different motor imagery tasks, namely the imagination of movement of the left hand (class 1), right hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on different days were recorded for each subject. Each session is comprised of 6 runs separated by short breaks. One run consists of 48 trials (12 for each of the four possible classes), yielding a total of 288 trials per session.

The subjects were sitting in a comfortable armchair in front of a computer screen. At the beginning of a trial ( t = 0 s), a fixation cross appeared on the black screen. In addition, a short acoustic warning tone was presented. After two seconds ( t = 2 s), a cue in the form of an arrow pointing either to the left, right, down or up (corresponding to one of the four classes left hand, right hand, foot or tongue) appeared and stayed on the screen for 1.25 s. This prompted the subjects to perform the desired motor imagery task. No feedback was provided. The subjects were ask to carry out the motor imagery task until the fixation cross disappeared from the screen at t = 6 s.

Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm) were used to record the EEG. All signals were recorded monopolarly with the left mastoid serving as reference and the right mastoid as ground. The signals were sampled with. 250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of the amplifier was set to 100 μV . An additional 50 Hz notch filter was enabled to suppress line noise

### References
.. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C., Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G. and Nolte, G., 2012. Review of the BCI competition IV. Frontiers in neuroscience, 6, p.55. """

# Results
To reproduce the results, use the main.ipnyb file.

 Among the 3 different classifiers tested, the one which outperformed the rest is the SVM classifier, with mean accuracy across subjects equal to 80.63%.
 
 ![image](https://user-images.githubusercontent.com/87859006/229521061-d8bf0efe-b98f-4bf8-9059-b4d495ecc533.png)



