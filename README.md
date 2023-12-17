# A Spatio-Temporal Probabilistic Neural Network to Forecast COVID-19 Counts

This code repository is related to the paper *A Spatio-Temporal Probabilistic Neural Network to Forecast COVID-19 Counts* by Ravenda et al. <br><br>



<img src="https://github.com/Fede-stack/Probabilistic-COVID19/blob/main/images/PNN.png" alt="Skeleton of the Probabilistic Neural Network architecture." width="500">

**Figure 1. Skeleton of the Probabilistic Neural Network architecture.**



To use the model, refer to the function:
- `model/MHCNN_poisson.py`: base Probabilistic Neural Network Model (as represented in Figure 1.) used within the paper.

Main steps for the preprocessing step are listed in:
- `Preprocessing/preprocessing.py` describes the necessary steps to process the data to be passed to the architecture.


For **citing** this work refer to

```
@article{ravendaspatio,
title={A Spatio-Temporal Probabilistic Neural Network to Forecast COVID-19 Counts},
author={Ravenda, Federico and Cesarini, Mirko and Peluso, Stefano and Mira, Antonietta}
}
```
