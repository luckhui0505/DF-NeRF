# DF-NeRF
DF-NeRF employs a unique approach to simulate the process of motion blur formation, thereby enhancing its ability to understand the blur formation process. We conducted experiments in both synthetic and real scenarios, and our method significantly improved PSNR, SSIM, and LPIPS metrics to the current state-of-the-art level, with the most significant improvement in LPIPS in particular. In synthetic scenes, LPIPS improves about 14.32\% on average compared to SOTA. We give experimental results in real scenarios, where LPIPS improves about 29.39\% compared to the benchmark model.

## Quick Start
### 1.Install environment
```
git clone https://github.com/luckhui0505/DF-NeRF.git
cd DF-NeRF
pip install -r requirements.txt
```

### 2. Download dataset
There are total of 31 scenes used in the paper. We mainly focus on camera motion blur and defocus blur, so we use 5 synthetic scenes and 10 real world scenes for each blur type. We also include one case of object motion blur. You can download all the data in [here](https://hkustconnect-my.sharepoint.com/personal/lmaag_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Flmaag%5Fconnect%5Fust%5Fhk%2FDocuments%2Fshare%2FCVPR2022%2Fdeblurnerf%5Fdataset&ga=1).
### 3. Setting parameters
Changing the data path and log path in the configs/demo_blurfactory.txt


## Method Overview
![image](https://github.com/luckhui0505/DF-NeRF/blob/main/framework.jpg) 
The overall network structure of DF-NeRF.
## Comparison of Experimental Results
![image](https://github.com/luckhui0505/DF-NeRF/blob/main/result1.jpg) 
Quantitative results on synthetic scenes. 
![image]([https://github.com/luckhui0505/DF-NeRF/blob/main/result2.jpg)) 
Quantitative results on real scenes. 
## Some Notes
### GPU Memory
We train our model on a RTX3090 GPU with 24GB GPU memory. If you have less memory, setting N_rand to a smaller value, or use multiple GPUs.
### Multiple GPUs
you can simply set <mark> num_gpu = <num_gpu> <mark> to use multiple gpus. It is implemented using <mark> torch.nn.DataParallel <mark>. We've optimized the code to reduce the data transfering in each iteration, but it may still suffer from low GPU usable if you use too much GPU.
## Acknowledge
This source code is derived from the famous pytorch reimplementation of NeRF, nerf-pytorch, Deblur-NeRF. We appreciate the effort of the contributor to that repository.
