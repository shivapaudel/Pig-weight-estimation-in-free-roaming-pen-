# Pig-weight-estimation-in-free-roaming-pen-
This repository contains the code used in our research on estimating the body weight of pigs using 3D point cloud data.

Paudel, Shiva, et al. "Deep learning models to predict finishing pig weight using point clouds." Animals 14.1 (2023): 31.

Abstract:

The selection of animals to be marketed is largely completed by their visual assessment,
solely relying on the skill level of the animal caretaker. Real-time monitoring of the weight of farm
animals would provide important information for not only marketing, but also for the assessment
of health and well-being issues. The objective of this study was to develop and evaluate a method
based on 3D Convolutional Neural Network to predict weight from point clouds. Intel Real Sense
D435 stereo depth camera placed at 2.7 m height was used to capture the 3D videos of a single
finishing pig freely walking in a holding pen ranging in weight between 20–120 kg. The animal
weight and 3D videos were collected from 249 Landrace × Large White pigs in farm facilities of the
FZEA-USP (Faculty of Animal Science and Food Engineering, University of Sao Paulo) between 5
August and 9 November 2021. Point clouds were manually extracted from the recorded 3D video
and applied for modeling. A total of 1186 point clouds were used for model training and validating
using PointNet framework in Python with a 9:1 split and 112 randomly selected point clouds were
reserved for testing. The volume between the body surface points and a constant plane resembling
the ground was calculated and correlated with weight to make a comparison with results from the
PointNet method. The coefficient of determination (R2 = 0.94) was achieved with PointNet regression
model on test point clouds compared to the coefficient of determination (R2 = 0.76) achieved from
the volume of the same animal. The validation RMSE of the model was 6.79 kg with a test RMSE of
6.88 kg. Further, to analyze model performance based on weight range the pigs were divided into
three different weight ranges: below 55 kg, between 55 and 90 kg, and above 90 kg. For different
weight groups, pigs weighing below 55 kg were best predicted with the model. The results clearly
showed that 3D deep learning on point sets has a good potential for accurate weight prediction even
with a limited training dataset. Therefore, this study confirms the usability of 3D deep learning on
point sets for farm animals’ weight prediction, while a larger data set needs to be used to ensure the
most accurate predictions.

