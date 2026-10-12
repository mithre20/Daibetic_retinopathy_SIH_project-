# Retinal Image Quality Assessment (IQA)

## What is Image Quality Assessment (IQA)?
Retinal Image Quality Assessment (IQA) is an automated safety gatekeeper powered by a dedicated MobileNetV3 neural network (`models/retinal_quality_mobilenet.pth`). It evaluates optical clarity, sharpness, illumination, contrast, and anatomical field visibility before permitting clinical screening.

## Why is IQA mandatory?
In rural outreach camps and telemedicine screening, low-cost cameras, patient movement, poor pupil dilation, cataract opacity, or dust on lenses frequently produce blurry or degraded images. Passing an unreadable image to a disease classifier can result in dangerous false negatives (missing life-threatening disease) or false positives. IQA guarantees that only diagnostically reliable images are analyzed.

## What are the IQA Quality Classes?
1. **Good Quality**: The optic disc, retinal blood vessels, and macula are crisp, well-illuminated, and clearly focused.
2. **Poor Quality**: The retina is partially obscured by moderate blur, under-exposure, severe shadow, or optical glare, making subtle microaneurysms or hemorrhages impossible to detect reliably.
3. **Unusable**: The image does not contain recognizable retinal structures, is completely black, out-of-focus, or depicts non-retinal artifacts.

## What is the Safety Threshold?
The default engineering threshold is **75% (0.75)** confidence for "Good Quality". Even if an image is predicted as Good Quality, if model confidence is below 75%, it is treated conservatively as BLOCKED to prevent borderline images from entering the screening pipeline.

## Why was my image rejected?
Your image was rejected because the IQA gate detected optical defects such as:
- Motion blur from patient movement or camera instability.
- Inadequate illumination (underexposed or too dark).
- Corneal or lens reflection/glare obscuring the macular region.
- Incomplete pupil alignment where the retinal field is vignetted or truncated.

## What happens when an image is rejected?
The screening pipeline is immediately halted. The DR prediction, risk category, and referral score are set to `null` (BLOCKED). A notification is displayed instructing the health worker to retake the scan.

## IMPORTANT SAFETY RULE: Rejection Does NOT Mean Healthy Eye
When an image is rejected, the system will **NEVER** state:
"Your eye is healthy." or "No diabetic retinopathy detected."
Instead, the system explicitly explains:
"The image quality was not sufficient for reliable screening. Please capture another image."
A rejected image cannot prove the presence or absence of diabetic retinopathy.

## Why does the system stop screening?
Allowing an AI model to evaluate a blurry or dark scan could produce a false sense of security (a false negative), leading a patient with severe retinopathy to miss urgent medical treatment. Stopping the screening protects patient safety.

## How can I capture a better retinal image?
1. Ask the patient to sit comfortably and fixate on the camera target light.
2. Ensure the room lighting is dimmed if using non-mydriatic cameras.
3. Hold the fundus camera completely steady or use a chinrest mount.
4. Adjust camera distance until the retinal illumination is uniform and glare rings are eliminated.
5. Re-focus specifically on the retinal vessels and optic disc before taking the photo.
