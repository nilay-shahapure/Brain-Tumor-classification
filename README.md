# Brain-Tumor-classification
End to End pipeline on a kaggle dataset for classifying brain tumors with a few advanced statistics/ analysis points




This repo contains:

- **train_advanced.py**  
  Full pipeline: train ResNet-18, MC-Dropout, Grad-CAM, UMAP/t-SNE.
- **results/**  
  Pre-computed artifacts so you can inspect outcomes without re-running.
- **data/**  
  Empty placeholder. To reproduce, place your Kaggle `train/` & `val/` folders here.

## Quick Look

- **Best accuracy:** shown in `results/best.pth`  
- **Uncertainty array:** `results/uncertainties.npy`  
- **Grad-CAM samples:** `results/gradcam_0.png` … `gradcam_3.png`  
- **Latent embedding:** `results/embedding_t-SNE.png`

## To Reproduce

1. Clone this repo.  
2. Install deps:  requirements.txt


3. Add your `data/train/yes`, `data/train/no`, `data/val/yes`, `data/val/no` folders under `data/`.  
4. Run:  python train_advanced.py --data_dir data --epochs 20 --batch_size 32 --lr 1e-4 --out_dir results --mc_T 20

