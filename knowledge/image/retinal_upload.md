# Retinal Fundus Image Upload Guidelines

## How do I upload an eye image?
1. Go to the **AI Pipeline** tab (Station 05: 5-Stage Explainable AI Pipeline).
2. Locate the **Retinal Fundus Scan** dropzone card in Step 1.
3. Drag and drop your image file directly into the dotted area, or click anywhere inside the box to open your file browser.
4. Select the fundus image. Once selected, an interactive preview showing the image dimensions, file name, and eye orientation will appear.

## Where do I upload the retinal image?
Uploads are processed on the **5-Stage Explainable AI Pipeline** page (`index_pipeline_original.html` or Station 05 on the master portal).

## What image formats are supported?
The system accepts standard digital raster formats:
- JPEG / JPG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- WebP (`.webp`)
- TIFF / TIF (`.tif`, `.tiff`)

## What is the maximum file size?
The recommended file size is between 500 KB and 15 MB. Files up to 25 MB are accepted by the server. Large clinical images are automatically resized and normalized during preprocessing.

## Why did my upload fail?
Uploads may fail due to:
1. **Unsupported file format**: Ensure you are uploading a valid image (JPG, PNG) and not a PDF, Word document, or video.
2. **Corrupt file**: The image file may be truncated or unreadable.
3. **Empty payload**: Ensure a file was actually selected before clicking Run.
4. **Network disconnection**: Check if the local backend server is running and accessible on port 5000.

## Can I replace an uploaded image?
Yes. Click the **Change Image** button beneath the preview canvas, or drag a new image file into the dropzone. The previous scan will be replaced immediately.

## Can I delete an uploaded image?
Yes. Clicking **Change Image** clears the preview and unloads the current image from memory.

## Can I upload multiple images at the same time?
The pipeline currently evaluates one retinal fundus scan per inference pass to maintain strict traceability and eye-specific attribution (Right Eye OD or Left Eye OS). To screen both eyes of a patient, screen the Right Eye (OD) first, review the results, and then upload the Left Eye (OS).

## Can I upload a normal eye photograph or selfie?
**No.** A standard smartphone selfie, external eye photo, or portrait photograph shows only the eyelids, sclera, iris, and pupil. It does NOT show the retina (the inner lining at the back of the eye). The deep learning models require a specialized optical fundus photograph captured using a fundus camera or ophthalmoscope adapter.

## What kind of retinal image is required?
The system requires a posterior pole or macular-centered digital retinal fundus photograph showing the optic disc, macula, fovea, and major retinal vascular arcades.

## What happens after uploading the image?
When you click **Run Screening Pipeline**, the system immediately passes the image through Stage 1: Retinal Image Quality Assessment (IQA).
- If the image is rated **Good Quality** and meets the safety threshold (>= 75% confidence), the system proceeds to Stage 2 (DR Classification), Stage 3 (Grad-CAM & SHAP Explainability), Stage 4 (U-Net Lesion Segmentation), and Stage 5 (Referral Triage).
- If the image is rated **Poor Quality** or **Unusable**, screening is blocked immediately with clear recapturing advice.
