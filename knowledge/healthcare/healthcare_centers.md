# Healthcare Centers and Eye Hospitals

## Finding Healthcare Facilities
The application maintains a verified registry of affiliated Primary Health Centers (PHCs), District General Hospitals, and Regional Institutes of Ophthalmology for patient referral triage.

## Verified Configured Centers
In the current outreach deployment configuration, the primary referral centers include:

1. **Thirunallur Primary Health Center (PHC)**
   - Type: Primary Screening Node
   - Focus: Frontline retinal photography, initial IQA gating, community triage.

2. **Kallakurichi Rural Mobile Unit**
   - Type: Mobile Telemedicine Screening Van
   - Focus: Field outreach camps, point-of-care report analysis.

3. **District General Hospital — Department of Ophthalmology**
   - Type: Secondary Care Referral Center
   - Services: Dilated slit-lamp biomicroscopy, fundus fluorescein angiography (FFA), macular optical coherence tomography (OCT), and medical retina clinics.

## Important Rule on Center Information & Availability
If a user inquires about a specific local hospital, ophthalmologist, opening hours, or private clinic not listed in the verified registry:
- The system **NEVER** invents or hallucinates facility names, addresses, doctor names, phone numbers, or appointment schedules.
- If information for a requested district or city is not configured in the system, the chatbot explicitly states:
  "Specific contact details or appointment schedules for that healthcare facility are not currently configured in the AI DR Screen system. Please contact your local Primary Health Center (PHC) coordinator, District Hospital helpdesk, or government health helpline for local referral appointments."

## What Should I Take to the Appointment?
When referred to an ophthalmologist or hospital:
1. A printed copy of your AI DR Screen summary report (including DR grade and explainability findings).
2. Recent medical and laboratory reports (especially blood glucose, HbA1c, and BP records).
3. A list of current medications (insulin, oral hypoglycemics, antihypertensives).
4. Government health identity card (e.g. ABHA / Ayushman Bharat card).
5. Sunglasses, as your pupils may be dilated with eye drops during the clinical examination.
