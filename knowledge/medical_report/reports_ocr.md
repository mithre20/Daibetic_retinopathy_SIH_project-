# Medical Laboratory Report Upload and OCR Analysis

## What Medical Reports Can I Upload?
AI DR Screen accepts diagnostic laboratory test reports, outpatient discharge summaries, and blood work documents that include glycemic indices and blood pressure records.

## What File Formats Are Supported for Medical Reports?
- Digital PDF documents (`.pdf`)
- Scanned laboratory report photographs or smartphone captures in JPEG, JPG, or PNG (`.jpg`, `.jpeg`, `.png`)

## What is OCR?
OCR stands for Optical Character Recognition. The application uses specialized OCR engines (including PyPDF for digital documents and Tesseract/EasyOCR for scanned images) to read text, find relevant clinical test names, and parse measured numeric values along with their units (e.g. mg/dL, %, mmHg).

## What Happens if OCR Fails or Makes a Mistake?
If a laboratory report is hand-written, faint, creased, or formatted unconventionally, the automated OCR extraction might fail to identify some values or extract a distorted number.
- The system flags unparsed values with a clear alert.
- **Manual Entry & Override**: Users and health staff can directly type or edit the glucose (FBS, RBS, PPBS), HbA1c, and systolic/diastolic blood pressure values directly into the input fields on the screen.
- Manual entries take immediate priority and are merged into the final clinical screening summary.

## Can I Upload an Older Report?
Recent laboratory reports (within the last 30 to 90 days) are recommended to reflect the patient's current glycemic status. If an older report is uploaded, the health worker should document the test date.

## Can I Upload Multiple Reports?
The application processes one primary laboratory report at a time per screening session to ensure unambiguous data attribution.
