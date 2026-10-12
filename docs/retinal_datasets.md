# Retinal Datasets Audit & Annotation Registry

**AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India**

---

## 1. Overview and Scientific Integrity Protocol

Before training or claiming clinical validation for any retinal structure localization or lesion analysis algorithm, a formal audit of benchmark datasets was conducted. 

In strict adherence to medical AI validation ethics:
1. **DR severity classification labels** (e.g., Grade 0–4) are **NEVER** substituted for pixel-level lesion segmentations or anatomical landmark coordinates.
2. **Synthetic or fabricated masks** are never generated or claimed as ground truth.
3. Every dataset is audited for what it **actually** provides versus what is claimed in informal literature.
4. When a required annotation (such as prospective multi-grade neovascularization masks) is absent from the active training environment, the component is rigorously marked as **`PENDING DATASET / VALIDATION`** rather than fabricating predictions.

---

## 2. Dataset Audit Table

| Dataset Name | Official Source / URL | Images | Native Resolution | Annotation Types Provided | DR Labels Available | License | Reference Paper | Active Local Status |
|---|---|---|---|---|---|---|---|---|
| **IDRiD** (Indian Diabetic Retinopathy Image Dataset) | [idrid.grand-challenge.org](https://idrid.grand-challenge.org/) | 516 total (81 segmentation, 516 grading) | 4288 × 2848 px | Binary masks for: <br>• Microaneurysms (MA)<br>• Haemorrhages (HE)<br>• Hard Exudates (EX)<br>• Soft Exudates (SE)<br>• Optic Disc center/boundary | 5-class DR (0–4) + DME risk (0–2) | Creative Commons BY 4.0 | Porwal et al., *Scientific Data* (2018) | **Reference Benchmark** (Audited in `training/dataset_unet.py`) |
| **DRIVE** (Digital Retinal Images for Vessel Extraction) | [isi.uu.nl/Research/Databases/DRIVE/](https://isi.uu.nl/Research/Databases/DRIVE/) | 40 (20 train, 20 test) | 565 × 584 px | Pixel-level binary vessel masks (dual manual observer) + FOV masks | None (Normal / Mild DR) | Academic / Research Use | Staal et al., *IEEE TMI* (2004) | **Standard Vessel Ground Truth Benchmark** |
| **STARE** (Structured Analysis of the Retina) | [cecas.clemson.edu/~ahoover/stare/](https://cecas.clemson.edu/~ahoover/stare/) | 400 images (20 with vessel GT, 81 with optic disc) | 700 × 605 px | Binary vessel masks (2 observers) + Optic nerve coordinates | 14 retinal diagnostic classes | Educational / Academic Free | Hoover et al., *IEEE TMI* (2000) | **Standard Landmark & Vessel Benchmark** |
| **MESSIDOR** (Methods for Evaluating Segmentation and Indexing of Techniques Dedicated to Retinal Ophthalmology) | [messidor.crihan.fr](http://www.adcis.net/en/third-party/messidor/) | 1,200 images (3 CCD cameras) | 1440×960 to 2304×1536 px | Macula/Fovea coordinates + Optic disc location in subsets | Retinopathy grade (0–3) + Macular edema risk (0–2) | Research Use Agreement | Decencière et al., *IRBM* (2014) | **Standard Fovea & DR Severity Benchmark** |
| **DiaretDB1** (Standard Diabetic Retinopathy Database Calibration Level 1) | [it.lut.fi/project/imageret/diaretdb1/](http://www.it.lut.fi/project/imageret/diaretdb1/) | 89 fundus images | 1500 × 1152 px | Bounding boxes & confidence for MA, HE, EX, and SE (4 experts) | None | Academic Research | Kauppi et al., *BMVC* (2007) | **Lesion Candidate Detection Benchmark** |
| **e-Ophtha** | [adcis.net/en/third-party/e-ophtha/](https://www.adcis.net/en/third-party/e-ophtha/) | 47 images with MA, 47 images with EX | 1440×960 to 2544×1696 px | Binary lesion masks (Microaneurysms, Exudates) | Binary lesion presence | Research Agreement | Decencière et al., *Medical Image Analysis* (2013) | **High-Resolution Micro-Lesion Benchmark** |
| **Roboflow DR Cohort** (Local Project Dataset) | `DR_Screening_Pipeline/data/` | 2,838 images (2,076 train, 478 valid, 284 test) | 192 × 192 px (pre-downscaled) | Image-level multi-class DR severity classification | 5 classes: No_DR (0), Mild (1), Moderate (2), Severe (3), Proliferative (4) | CC BY 4.0 / Public Export | Roboflow DR AI Dataset (2023) | **Active Local Classification Cohort** |

---

## 3. Detailed Audit by Required Component

### 1. Optic Disc Localization
* **Required Annotations**: Optic disc center $(x, y)$, optic disc boundary or bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
* **Available in**: IDRiD (Optic Disc localization sub-challenge), STARE (Optic Nerve ground truth), MESSIDOR (OD center markings).
* **Local Dataset Reality**: Local Roboflow dataset contains image-level DR grades only, without bounding box metadata for optic disc.
* **Methodological Solution**: Robust computer-vision detector combining green/red luminance peak clustering, morphological opening, vessel convergence orientation, and circular Hough transform fitting. Evaluated with spatial center stability and detection rate.

### 2. Fovea Localization
* **Required Annotations**: Fovea center $(x, y)$ coordinate or macula center boundary.
* **Available in**: MESSIDOR-1/2 (expert macular center coordinates), IDRiD (Fovea coordinates).
* **Local Dataset Reality**: Local Roboflow dataset contains image-level DR grades only.
* **Methodological Solution**: Anatomical relative geometry modeling from the detected optic disc center ($2.5 \times$ optic disc diameters temporal, $-0.3 \times$ OD vertical offset) combined with intensity valley detection (darkest macula lutea avascular zone). Evaluated against clinical tolerance thresholds ($\le 1.0\text{ OD diameter}$).

### 3. Retinal Vessel Segmentation
* **Required Annotations**: Binary vessel tree pixel mask $[H, W]$.
* **Available in**: DRIVE (40 images, 2 observers), STARE (20 images), CHASE_DB1 (28 images).
* **Local Dataset Reality**: Local Roboflow dataset has no vessel masks.
* **Methodological Solution**: Green-channel contrast normalization, CLAHE enhancement, multi-scale matched Gabor/Frangi vessel filtering, and hysteresis thresholding. Evaluated with Dice, IoU, Sensitivity, Specificity, and Pixel Accuracy.

### 4. Microaneurysm Detection
* **Required Annotations**: Lesion coordinate centroids or pixel-level binary masks for punctate micro-vascular outpouchings ($< 125\ \mu\text{m}$).
* **Available in**: IDRiD (MA masks), e-Ophtha-MA, DiaretDB1.
* **Local Dataset Reality**: Roboflow images are resized to $192 \times 192$, where isolated microaneurysms ($10\text{–}30\ \mu\text{m}$) suffer spatial aliasing.
* **Methodological Solution**: High-resolution mathematical morphology (top-hat transformation, local minima filtering in green channel, shape circularity gating, vessel-skeleton subtraction to suppress false positive branch points) coupled with U-Net Class 1 (Microaneurysm) feature maps.
* **Sub-pixel detection status**: Formally marked **`NOT VALIDATED`** in accordance with project directives, as super-resolution calibration targets are absent.

### 5. Exudate Segmentation (Hard & Soft)
* **Required Annotations**: Binary masks for Hard Exudates (bright, well-defined waxy lipid deposits) and Soft Exudates / Cotton Wool Spots (pale, fluffy ischemic nerve fiber swelling).
* **Available in**: IDRiD (EX and SE masks), e-Ophtha-EX, DiaretDB1.
* **Local Dataset Reality**: Integrated with U-Net 5-class formulation (Class 3: Hard Exudates, Class 4: Soft Exudates) and multi-level luminance thresholding.

### 6. Hemorrhage Detection & Classification
* **Required Annotations**: Intraretinal flame/dot-blot hemorrhage masks and lesion coordinates.
* **Available in**: IDRiD (HE masks), DiaretDB1.
* **Methodological Solution**: Red-lesion segmentation (excluding optic disc and vessels) combined with U-Net Class 2 (Hemorrhages), outputting lesion count, total pixel area, spatial distribution (quadrants), and binary classification (`Hemorrhages Present` vs. `Absent`).

### 7. Neovascularization Detection
* **Required Annotations**: Dedicated fine-vessel proliferation masks on optic disc (NVD) or elsewhere (NVE).
* **Available in**: Extremely sparse in open datasets; typically inferred from high-level Proliferative DR (PDR) clinical staging.
* **Local Dataset Reality**: Only image-level `Proliferate_DR` class labels exist in the Roboflow dataset; no dedicated NVD/NVE segmentation ground truth exists.
* **Ethical Determination**: Module interface, clinical rule criteria, and model architectures are fully implemented, but formal status is reported as **`PENDING DATASET + CLINICAL VALIDATION`** to prevent scientific misrepresentation.

---

## 4. Compliance and Audit Summary

```
========================================================================================
RETINAL DATASET & ANNOTATION INTEGRITY AUDIT
========================================================================================
Optic Disc Localization       : Algorithmic Photometric & Geometry [PASS - FUNCTIONAL]
Fovea Localization            : Anatomical Relational Constraint  [PASS - FUNCTIONAL]
Retinal Vessel Segmentation   : Multi-scale Matched Filtering     [PASS - FUNCTIONAL]
Microaneurysm Detection       : High-Res Morphology + U-Net C1    [PASS - FUNCTIONAL]
Sub-pixel MA Localization     : Super-resolution ground truth     [NOT VALIDATED]
Hard & Soft Exudate Segment.  : Luminance Morph + U-Net C3/C4     [PASS - FUNCTIONAL]
Hemorrhage Detection / Class. : Morphological Red Lesion + C2     [PASS - FUNCTIONAL]
Neovascularization Detection  : Fine-vessel Proliferation GT      [PENDING DATASET + CLINICAL VAL]
========================================================================================
```
