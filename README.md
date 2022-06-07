# implementation of "Fast Unsupervised Bayesian Image Segmentation With Adaptive Spatial Regularisation" (Marcelo Pereyra and Steve McLaughlin)

## Getting started

Run the following command to install the dependecies:

```
pip install -r requirements.txt
```

For examples of usage please refer to the ```segmentation.py``` script.

## Results

| Implementation/Image | GMM4  | GMM8  | LMM2  | PMM3  |
| --------- | ----- | ----- | ----- | ----- |
| Paper     | 88.9% | 89.6% | 96.0% | 93.1% |
| Mine      | 95.6% | 87.8% | 93.3% | 96.5% |

![](results/GMM4_results.png)
![](results/GMM8_results.png)
![](results/LMM2_results.png)
![](results/PMM3_results.png)
![](results/Bacteria_results.png)
![](results/Brain_results.png)
![](results/Lungs_results.png)
![](results/SAR_results.png)