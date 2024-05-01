# A probabilistic spatio-temporal neural network to forecast COVID-19 counts
This code repository is related to the paper *A Spatio-Temporal Probabilistic Neural Network to Forecast COVID-19 Counts* by Ravenda et al. <br><br>



<img src="https://github.com/Fede-stack/Probabilistic-COVID19/blob/main/images/PNN.png" alt="Skeleton of the Probabilistic Neural Network architecture." width="500">

**Figure 1. Skeleton of the Probabilistic Neural Network architecture.**

To use the model, refer to the function:
- `model/MHCNN_poisson.py`: base Probabilistic Neural Network Model (as represented in Figure 1.) used within the paper.

Main steps for the preprocessing step are listed in:
- `Preprocessing/preprocessing.py` describes the necessary steps to process the data to be passed to the architecture.


<img src="https://github.com/Fede-stack/Probabilistic-COVID19/blob/main/images/embeddings.png" alt="" width="500">

**Figure 2. Low-Dimensional representations (Embeddings) related to the spatial information input.**

To implement Entity Embedding for both Spatial and Temporal Information refer to: 
- `entity_embedding.py` 
  
For **citing** this work refer to

```
@article{ravenda2024probabilistic,
  title={A probabilistic spatio-temporal neural network to forecast COVID-19 counts},
  author={Ravenda, Federico and Cesarini, Mirko and Peluso, Stefano and Mira, Antonietta},
  journal={International Journal of Data Science and Analytics},
  pages={1--8},
  year={2024},
  publisher={Springer}
}
```
