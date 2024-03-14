# Assignment S7

## Step1

- **Targets**: Tried a few iterations on setup, skeleton and an initial lighter model targetting params < 8k. 
  - Models used in this step:
    - model_1_setup
    - model_2_skeleton (Model skeleton included using 3 conv blocks (8conv all together) separated by maxpooling. First conv block had a receptive field of 5, followed by second block which had a receptive field of ~14, followed by last block that has a receptive field of 28. (Trying to align the edges, patterns, full image across each conv block)
    - model_3_lighter_model
- **Results:**
  - Setup: model_1_setup
    - Parameters: **6.3M**
    - Best training accuracy: **99.83**
    - Best test accuracy: **99.11**
  - Skeleton: model_2_skeleton
    - Parameters: **57.8k**
    - Best training accuracy: **99.01**
    - Best test accuracy: **98.9**
  - Lighter model: model_3_lighter_model
    - Parameters: **7.7k**
    - Best training accuracy: **98.7**
    - Best test accuracy: **98.6**
- **Analysis:**
  - Inital setup (model_1_setup) had a huge model 6.3M, but was primarily used for setting up entire pipeline.  Train/Test accuracy of ** 99.83 / 99.11 ** Huge model, overfitting.
  - Skeleton model (model_2_skeleton).  Train/Test accuracy of ** 99.01 / 98.9 ** Relatively large model much higher than the 8k target, lowered train accuracy and slightly (not a lot) of overfitting.
  - Lighter model model had 7690 params.  Train/Test accuracy of ** 99.01 / 98.9 ** Good starting point for the model, lot's of scope for improvement given only 99% train accuracy.


## Step2

- **Targets:** Tried a few iterations on batch normalization (to improve model accuracy), regularization (to prevent overfitting) and adding a GAP layer near the output. 
  - Models used in this step:
    - model_4_batchnorm
    - model_5_regularization ((by adding a dropout layer after one of the conv layer)
    - model_6_gap
- **Results:**
  - Batchnorm: model_4_batchnorm
    - Parameters: **7830**
    - Best training accuracy: **99.71**
    - Best test accuracy: **98.96**
  - Regularization: model_5_regularization 
    - Parameters: **7830**
    - Best training accuracy: **99.48**
    - Best test accuracy: **98.92**
  - Adding GAP layer near output: model_6_gap
    - Parameters: **7430**
    - Best training accuracy: **99.37**
    - Best test accuracy: **98.97**
- **Analysis:**
  - Batchnorm (model_4_batchnorm) improved training accuracy as expected (99.7%), and slightly improved the test accuracy closer to 99%. Model overfitting even now. Tried to add regularization.
  - regularization model (model_5_regularization) reduced the gap between test and train accuracy, slightly degraded the train accuracy as expected.
  - Adding GAP at the output (model_6_gap) reduced the model size further, reduced train accuracy slightly (negligible) but did not decrease the test accuracy. Gives the model additional wiggle room for playing around with more layers later and squueze performance further.



## Step3
- **Targets:** Tried a few iterations on batch normalization (to improve model accuracy), regularization (to prevent overfitting) and adding a GAP layer near the output. 
  - Models used in this step:
    - model_7_max_capacity (added another conv layer at the end to increase model capacity to max)
    - model_8_dropout_maxpool (added dropout in each conv block, played around with dropout value, optimal was found to be at 0.01, implying not a lot of overfitting in the existing model, as is evident from the previous steps where the gap between test/train was not high)
    - image augmentation and scheduler + learning rates (added image augmentation and played around with scheduler + LR) to improve model performance on the whole. 
- **Results:**
  - Increase model capacity: model_7_max_capacity
    - Parameters: **7914**
    - Best training accuracy: **99.42**
    - Best test accuracy: **99.3**
  - Adjust maxpool and dropout locations/values: model_8_dropout_maxpool 
    - Parameters: **7914**
    - Best training accuracy: **99.4**
    - Best test accuracy: **99.3**
  - Image augmentation and scheduler+LR: 
    - Parameters: **7914**
    - Best training accuracy: **99.75**
    - Best test accuracy: **99.43**
- **Analysis:**
  - Increase model capacity (model_7_max_capacity) close to the 8k limit, improved test/train accuracy slightly.
  - Dropout value set to 0.01 on each conv (model_8_dropout_maxpool) as the model wasn't really overfitting in previous stages. This did not help the model a lot.
  - Image augmentation improved train acuracy from 99.4% to 99.8% . This helped to improve test accuracy to 99.43%, meeting our accuracy requiremetn. Played around with LR and scheduler to expedite training/test accuracy, and got to the required accuracy within 6-8 epochs as seen in the last notebook.
