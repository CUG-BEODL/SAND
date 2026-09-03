# SAND: Sub-Annual Net for Land Dynamics 🌍📡

This repository provides the official implementation of **SAND (Sub-Annual Net for Land Dynamics)**, a Transformer-based autoregressive framework for fine-scale, sub-annual land cover mapping using time-series remote sensing imagery.

Traditional land cover mapping approaches commonly generate annual products and therefore have limited capability to characterize land cover transitions occurring within a single year. SAND explicitly models temporal semantic evolution from dense Sentinel-2 time series, enabling more detailed characterization of intra-annual land dynamics and improved identification of short-term land cover transitions.

Experiments conducted across multiple cities demonstrate the effectiveness of SAND for high-temporal-resolution land cover mapping and temporal semantic change detection.

![SAND Framework](https://github.com/user-attachments/assets/48be0168-04fd-4982-8417-a390515ca6f3)

📄 **Paper:**
*Transformer-based temporal semantic change detection for intra-annual land cover dynamics from Sentinel-2 time series*

---

## Usage 🚀

### Training 🏋️‍♂️

To train SAND, run:

```bash
python train.py
```

Please configure the dataset paths, training parameters, and other model settings in the corresponding configuration files before training.

### Testing 🧪

To evaluate a trained SAND model, run the corresponding testing script with the trained model checkpoint and test dataset.

---

## Dataset 📊

All datasets used in this study are publicly available.

The datasets can be downloaded from:

[Google Drive Dataset](https://drive.google.com/drive/folders/1VkQ3q0m4LfMT4YwA0HkuOCvsByqtDAyC?usp=sharing)

---

## Results 🏆

SAND has been evaluated across multiple cities and demonstrates strong capability for identifying land cover transitions occurring within a single year.

Compared with conventional annual-scale land cover mapping approaches, SAND provides finer temporal characterization of land dynamics and can effectively capture short-term and subtle semantic transitions from Sentinel-2 time-series observations.

![SAND Results](https://github.com/user-attachments/assets/17977fba-6d25-43e1-80ec-d616cdd8171b)

### Key Features

* ✅ **Transformer-based temporal modeling** for learning long-term dependencies from remote sensing time series.
* ✅ **Autoregressive decoding** for sequential prediction of land cover semantic states and temporal transitions.
* ✅ **Sub-annual land cover mapping** for characterizing land dynamics at finer temporal scales.
* ✅ **Temporal semantic change detection** for identifying short-term land cover transitions within a single year.
* ✅ **Sentinel-2 time-series support** for high-spatial- and high-temporal-resolution land surface monitoring.

---

## Citation 📚

If you find SAND useful in your research, please cite the following paper:

```bibtex
@article{long2026transformer,
  title     = {Transformer-based method for detecting temporal semantic changes in land cover dynamics within a single year, using Sentinel-2 time series data},
  author    = {Long, Xu and Liu, Lirong and He, Haixu and Sun, Haonan and Yan, Jining},
  journal   = {Remote Sensing Letters},
  volume    = {17},
  number    = {12},
  pages     = {1530--1542},
  year      = {2026},
  publisher = {Taylor \& Francis}
}
```

---

## Acknowledgements 🙏

We gratefully acknowledge the **TSSCD** project for providing valuable methodological and implementation references for time-series semantic change detection.

TSSCD repository:

https://github.com/CUG-BEODL/TSSCD

---

## Contact 📬

For questions, suggestions, or research collaborations, please contact:

📧 **Email:** [20161001925@cug.edu.cn](mailto:20161001925@cug.edu.cn)

We welcome discussions and collaborations related to land cover mapping, remote sensing time-series analysis, and temporal semantic change detection.
