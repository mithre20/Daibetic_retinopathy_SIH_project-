# U-Net Retinal Lesion Segmentation

## What is U-Net?
U-Net is an encoder-decoder convolutional neural network architecture with skip connections specifically developed for biomedical image segmentation. In AI DR Screen, U-Net performs pixel-level semantic segmentation across the retinal fundus image (`models/unet_model.py`).

## Why is U-Net used?
Classification models (like EfficientNet) can only provide an image-level label (e.g. "Moderate DR"), but clinicians need to know the exact physical location, count, and surface area of individual lesions. U-Net bridges this gap by generating precise pixel masks of diabetic lesions.

## What does Segmentation mean?
Segmentation means labeling every single pixel in the 512x512 image with its anatomical or pathological class, delineating the exact contours and spatial distribution of retinal lesions.

## What Lesions can U-Net Identify?
U-Net segments 4 key microvascular lesion categories:
1. **Microaneurysm (Class 1)**: Tiny, isolated outpouchings of weakened retinal capillaries. They appear as small circular red dots (< 125 microns). Earliest clinical hallmark of non-proliferative diabetic retinopathy.
2. **Hemorrhage (Class 2)**: Intraretinal bleeding caused by ruptured capillaries or microaneurysms. Includes dot-and-blot hemorrhages (in deeper retinal layers) and flame-shaped hemorrhages (in the superficial nerve fiber layer).
3. **Hard Exudate (Class 3)**: Bright yellow, waxy-looking lipid deposits that leak from chronically damaged retinal vessels. When forming a circinate ring near the fovea, they indicate diabetic macular edema (DME).
4. **Soft Exudate / Cotton-Wool Spot (Class 4)**: Fluffy, white/grayish patches caused by micro-infarctions of the retinal nerve fiber layer due to local ischemia and axoplasmic stasis.

## Does U-Net independently determine the overall DR Grade?
**No.** U-Net provides anatomical lesion maps and lesion surface area metrics. The overall DR severity grading is determined by the primary classifier (EfficientNet-B0) and validated through the deterministic clinical referral engine. U-Net serves as transparent visual corroboration.
