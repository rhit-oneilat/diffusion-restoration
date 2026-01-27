# Diffusion-Based Face Restoration (CSSE463)

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for Image Restoration. This project aims to replicate and extend the findings of **RePaint** (Lugmayr et al.) to perform inpainting and enhancement on the CelebA dataset.

**Team:** Caleb Lehman, Rhys Phelps, Aidan O’Neil, Isaac Robbins

## Project Structure

```bash
.
├── data/                   # Data storage (GitIgnored)
│   └── celeba_raw/         # Unzipped CelebA images
├── src/                    # Model source code (U-Net, Scheduler)
├── notebooks/              # Jupyter experiments
├── prelim_check.py         # Baseline MSE verification script
├── requirements.txt        # Python dependencies
└── README.md

```

## Setup & Installation

### 1. Environment

This project is designed to run on the CSSE department servers (`gebru`, `noether`, etc.)

```bash
# Clone the repo
git clone [https://github.com/rhit-oneilat/diffusion-restoration.git](https://github.com/rhit-oneilat/diffusion-restoration.git)
cd diffusion-restoration

# Create a virtual env
python -m venv env
source env/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm

```

### 2. Data Download 

We use the **CelebA** dataset. Because the `torchvision` automatic download often fails due to Google Drive quotas, we must install it manually.

**Expected Path:**
The code expects images to be individual JPEGs located at:
`./data/celeba_raw/img_align_celeba/*.jpg`

#### Option A: If you are on the Rose-Hulman Server (`gebru`)

The data is likely already set up in our shared workspace. Link it to your repo:

```bash
# Make sure your local data folder exists
mkdir -p data

# Create a symbolic link to the shared copy (Saves space!)
ln -s /work/csse463/202520/06/data/celeba_raw data/celeba_raw

```

#### Option B: If you need to download it from scratch

1. Download `img_align_celeba.zip` from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
2. Upload/Move the zip file to `data/`.
3. Unzip and fix the folder nesting (Kaggle likes to nest folders inside folders):

```bash
mkdir -p data/celeba_raw
unzip img_align_celeba.zip -d data/celeba_raw

# FIX THE NESTING:
cd data/celeba_raw/img_align_celeba/img_align_celeba
mv *.jpg ..
cd ..
rmdir img_align_celeba

```

## Usage

### 1. Run Baseline Analysis

To verify the data is loaded correctly and check the "Do Nothing" baseline MSE vs a Simple CNN:

```bash
python prelim_check.py

```

*Current Baseline MSE Target: ~0.0099*

### 2. Training 

*Instructions for training the DDPM U-Net will be added in Week 8.*

## References

1. **Denoising Diffusion Probabilistic Models**, Ho et al. 2020.
2. **RePaint: Inpainting using Denoising Diffusion Probabilistic Models**, Lugmayr et al. 2022.
3. **ExposureDiffusion**, Wang et al. 2023.



