# Assignment 8


## - what is your code all about?
The repo for S8 contains 3 notebooks, each using a different normalization technique.
- BatchNorm: ERAV2_S8_BN.ipynb
- GroupNorm: ERAV2_S8_GN.ipynb
- LayerNorm: ERAV2_S8_LN.ipynb

The repo also has a models.py file that has the details of the 3 models used for each scenario.
- BatchNorm: Net_BN 
- GroupNorm: Net_GN (with 4 groups)
- LayerNorm: Net_LN

The networks have been optimized to be under 50k parameters, with a receptive field of 44? Details for the BatchNorm model are given below.
![Alt text](model_summary.png?raw=true "Title")

## - your findings for normalization techniques?
| Norm | Train accuracy | Test accuracy |
| :---         |     :---:      |          ---: |
| Batch Norm   | 72.3     | 73.8    |
| Group Norm     | 66.8       | 65.3      |
| Layer Norm    | 62       | 61.4      |


- BatchNorm performed the best in terms of training and test accuracy (>70%). Also, model had further scope of improving accuracy with additional epochs as we did not observe any overfitting on the model  after 20 epochs.
- GroupNorm performed slightly worse than BatchNorm (nno modifications done on top of the model to optimze grouNorm model or model hyperparameters, just replaced BN with GN). Observed some overfitting on the model, but could have optimized/finetuned the model further if needed.
- LayerNor performed the worst of the 3 models (no modifications on mode or hyperparameters, only changed GN to LN). Model overfitting and has lower train/test error relative to other 2 models.

## - add all your graphs
- Training and test errors in each normalization scenario. 
![Alt text](S8_image1.jpg?raw=true "Title")


## - your collection-of-misclassified-images
- Some errors from BatchNorm model below
![Alt text](bn_images.png?raw=true "Title")


- Some errors from GroupNorm model below
![Alt text](gn_images.png?raw=true "Title")


  
- Some errors from LayerNorm model below
![Alt text](ln_images.png?raw=true "Title")  
