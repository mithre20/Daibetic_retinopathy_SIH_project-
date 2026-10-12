import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root directory to sys.path so backend and training can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.inference import DRInferenceEngine
from backend.model.efficientnet_dr import get_device

app = Flask(__name__, static_folder=ROOT_DIR, static_url_path="")
CORS(app)

# Register Chatbot Assistant Modular Blueprint
from chatbot.api.chat_routes import chat_bp
app.register_blueprint(chat_bp)

# Initialize inference engine lazily or at startup
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = DRInferenceEngine()
    return engine

@app.route("/api/health", methods=["GET"])
def health_check():
    eng = get_engine()
    return jsonify({
        "status": "healthy",
        "service": "XAIRIS — Explainable AI for Diabetic Retinopathy Screening",
        "model": "EfficientNet-B0 (Transfer Learning)",
        "classes": 5,
        "device": str(eng.device),
        "weights_loaded": eng.weights_path is not None and os.path.exists(eng.weights_path)
    }), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Input: Retinal fundus image (file upload with key 'image' or 'file', or raw bytes)
    Output: JSON with prediction, severity, confidence, and 5-class probabilities
    """
    # Check if a file was uploaded in request.files
    file_obj = None
    if "image" in request.files:
        file_obj = request.files["image"]
    elif "file" in request.files:
        file_obj = request.files["file"]
    elif len(request.files) > 0:
        # Take the first uploaded file
        file_obj = next(iter(request.files.values()))

    # If not in files, check if sent in request.data
    input_data = file_obj if file_obj is not None else request.data

    if not input_data or (isinstance(input_data, bytes) and len(input_data) == 0):
        return jsonify({
            "error": "No retinal fundus image provided. Please upload an image file with key 'image' or 'file'."
        }), 400

    try:
        eng = get_engine()
        result = eng.predict(input_data)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal inference error: {str(e)}"}), 500


iqa_engine_instance = None
gatekeeper_instance = None

def get_iqa_engine():
    global iqa_engine_instance
    if iqa_engine_instance is None:
        from inference.predict_iqa import IQAInferenceEngine
        iqa_engine_instance = IQAInferenceEngine()
    return iqa_engine_instance

def get_gatekeeper():
    global gatekeeper_instance
    if gatekeeper_instance is None:
        from inference.screening_pipeline import RetinalScreeningGatekeeper
        gatekeeper_instance = RetinalScreeningGatekeeper()
    return gatekeeper_instance


@app.route("/api/iqa/predict", methods=["POST"])
def iqa_predict():
    """
    POST /api/iqa/predict
    Input: Retinal fundus image (file upload with key 'image' or 'file', raw bytes, or base64 JSON)
    Output: JSON with quality class, confidence, probabilities, screening_allowed flag, and recommendation
    """
    import base64
    import io
    import uuid

    raw_bytes = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                pass

    if raw_bytes is None:
        file_obj = None
        if "image" in request.files:
            file_obj = request.files["image"]
        elif "file" in request.files:
            file_obj = request.files["file"]
        elif len(request.files) > 0:
            file_obj = next(iter(request.files.values()))

        if file_obj is not None:
            raw_bytes = file_obj.read()
        elif request.data and len(request.data) > 0:
            raw_bytes = request.data

    if not raw_bytes:
        return jsonify({
            "error": "No retinal fundus image provided. Upload an image file with key 'image' or 'file', or JSON with 'image'."
        }), 400

    try:
        engine = get_iqa_engine()
        threshold = request.args.get("threshold", default=0.75, type=float)
        res = engine.predict(raw_bytes, threshold=threshold)

        # Optional MongoDB logging
        try:
            from chatbot.data.mongo_db import get_mongo_store
            store = get_mongo_store()
            screening_id = request.headers.get("X-Screening-ID", str(uuid.uuid4()))
            image_id = request.headers.get("X-Image-ID", f"img_{uuid.uuid4().hex[:8]}")
            store.record_iqa_screening(
                screening_id=screening_id,
                image_id=image_id,
                quality_class=res["prediction"],
                confidence=res["confidence"],
                screening_allowed=res["screening_allowed"]
            )
        except Exception:
            pass

        class_keys = ["good", "poor", "unusable"]
        class_id = res.get("class_id", 0)
        quality_key = class_keys[class_id] if class_id < len(class_keys) else "unknown"
        conf = res.get("confidence", 0.0)
        thresh = res.get("confidence_threshold", threshold)

        return jsonify({
            "quality_class": class_id,
            "quality_label": res.get("prediction", "Unknown"),
            "quality_key": quality_key,
            "confidence": conf,
            "confidence_pct": round(conf * 100.0, 2),
            "threshold": thresh,
            "is_confident": bool(conf >= thresh),
            "screening_allowed": res.get("screening_allowed", False),
            "status": res.get("status", "REJECTED"),
            "class_probabilities": res.get("probabilities", {}),
            "recommendation": res.get("recommendation", ""),
            "dimensions": res.get("dimensions", []),
            "auxiliary_metrics": res.get("quality_indicators", {})
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal IQA inference error: {str(e)}"}), 500


@app.route("/api/screen", methods=["POST"])
def screen_retinal_image():
    """
    POST /api/screen
    Complete Gated Screening Pipeline:
    1. MobileNetV3 IQA checks image quality.
    2. If Poor, Unusable, or Low Confidence -> SCREENING BLOCKED (returns recapture notice, dr_prediction=null).
    3. If Good Quality -> EfficientNet-B0 predicts DR Grade 0-4 + Referral Engine recommendations.
    """
    import base64

    raw_bytes = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                pass

    if raw_bytes is None:
        file_obj = None
        if "image" in request.files:
            file_obj = request.files["image"]
        elif "file" in request.files:
            file_obj = request.files["file"]
        elif len(request.files) > 0:
            file_obj = next(iter(request.files.values()))

        if file_obj is not None:
            raw_bytes = file_obj.read()
        elif request.data and len(request.data) > 0:
            raw_bytes = request.data

    if not raw_bytes:
        return jsonify({
            "error": "No retinal fundus image provided. Upload an image file with key 'image' or 'file'."
        }), 400

    try:
        gatekeeper = get_gatekeeper()
        threshold = request.args.get("threshold", default=0.75, type=float)
        result = gatekeeper.screen(raw_bytes, threshold=threshold)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal screening pipeline error: {str(e)}"}), 500



@app.route("/predict-segmentation", methods=["POST"])
@app.route("/api/predict-segmentation", methods=["POST"])
def predict_segmentation():
    """
    POST /predict-segmentation & POST /api/predict-segmentation
    Input: Retinal fundus image (file upload, raw bytes, or JSON {"image": "<base64/path>"})
    Output: JSON conforming to required contract:
    {
      "mask": "data:image/png;base64,...",
      "overlay": "data:image/png;base64,...",
      "lesions": {
        "microaneurysm": bool,
        "hemorrhage": bool,
        "hard_exudate": bool,
        "soft_exudate": bool
      }
    }
    """
    import base64
    import io
    from inference.predict_unet import predict_retinal_lesions

    raw_bytes = None

    # 1. Check JSON payload
    if request.is_json:
        data = request.get_json(silent=True) or {}
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                # strip data URI prefix
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                # check if it's a file path
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    # 2. Check request.files
    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    # 3. Check request.data
    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({
            "error": "No retinal fundus image provided. Send an image via file upload, raw body, or JSON {'image': 'base64'}."
        }), 400

    try:
        res = predict_retinal_lesions(image_input=raw_bytes, img_size=512)
        
        # Read generated mask and overlay as base64 data URLs
        with open(res["predicted_mask_path"], "rb") as f:
            mask_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        with open(res["lesion_overlay_path"], "rb") as f:
            overlay_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

        response_payload = {
            "mask": mask_b64,
            "overlay": overlay_b64,
            "lesions": res["detections"],
            "pixel_coverage": res["pixel_coverage_percentage"],
            "explanation": res["explanation"]
        }
        return jsonify(response_payload), 200
    except Exception as e:
        return jsonify({"error": f"Segmentation inference error: {str(e)}"}), 500


@app.route("/api/gradcam", methods=["GET", "POST"])
@app.route("/api/explain", methods=["GET", "POST"])
def explain_gradcam():
    """
    GET/POST /api/gradcam & /api/explain
    Generates Grad-CAM visual explanation for an uploaded retinal fundus scan.
    Parameters:
      - image: File upload, raw body, JSON {"image": "<base64/path>"}, or query param ?image=...
      - target_class: Optional integer (0..4). Defaults to model's predicted DR grade.
      - alpha: Heatmap overlay transparency in [0, 1] (default: 0.4).
      - with_unet: Optional boolean to include U-Net lesion segmentation comparison.
    """
    import base64
    import io
    import numpy as np
    from PIL import Image
    from inference.gradcam import (
        GradCAM,
        apply_colormap_and_overlay,
        calculate_gradcam_unet_overlap,
        pil_to_base64,
        MEDICAL_DISCLAIMER
    )
    from backend.model.inference import get_inference_transforms

    # 1. Parse Image Input
    raw_bytes = None
    target_class = None
    alpha = 0.4
    with_unet = False

    # Check query params
    if request.args.get("target_class") is not None:
        try:
            target_class = int(request.args.get("target_class"))
        except ValueError:
            target_class = None
    if request.args.get("alpha") is not None:
        try:
            alpha = float(request.args.get("alpha"))
        except ValueError:
            alpha = 0.4
    if request.args.get("with_unet") is not None:
        with_unet = request.args.get("with_unet").lower() in ["1", "true", "yes"]

    # Check JSON payload
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if "target_class" in data and target_class is None:
            try:
                target_class = int(data["target_class"])
            except (ValueError, TypeError):
                target_class = None
        if "alpha" in data:
            try:
                alpha = float(data["alpha"])
            except (ValueError, TypeError):
                pass
        if "with_unet" in data:
            with_unet = bool(data["with_unet"])

        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    # Check multipart files
    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    # Check raw data
    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    # Check GET query image parameter
    if raw_bytes is None and request.args.get("image"):
        img_param = request.args.get("image")
        if os.path.isfile(img_param):
            with open(img_param, "rb") as f:
                raw_bytes = f.read()
        elif os.path.isfile(os.path.join(ROOT_DIR, img_param)):
            with open(os.path.join(ROOT_DIR, img_param), "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({
            "error": "INPUT IMAGE NOT FOUND: Please upload an image file or provide a valid image parameter."
        }), 400

    try:
        # 2. Validate and load PIL image
        img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        eng = get_engine()
        device = eng.device
        model = eng.model

        # 3. Preprocess matching inference pipeline
        transform = get_inference_transforms()
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 4. Target Layer
        target_layer = model.get_gradcam_target_layer()
        layer_name = "backbone.features[8][0] (Conv2d: 320 -> 1280)"

        # 5. Generate Grad-CAM
        with GradCAM(model, target_layer) as gcam:
            cam_res = gcam.generate_cam(input_tensor, target_class=target_class)

        heatmap_raw = cam_res["heatmap"]
        pred_idx = cam_res["prediction"]
        pred_severity = cam_res["predicted_severity"]
        confidence = cam_res["confidence"]
        probs = cam_res["probabilities"]
        tgt_idx = cam_res["target_class"]
        tgt_severity = cam_res["target_severity"]
        sanity = cam_res["sanity_stats"]

        # 6. Apply colormap and overlay
        _, _, heatmap_pil, overlay_pil = apply_colormap_and_overlay(
            original_img=img_pil,
            heatmap=heatmap_raw,
            alpha=alpha,
            colormap_name="jet"
        )

        heatmap_b64 = pil_to_base64(heatmap_pil)
        overlay_b64 = pil_to_base64(overlay_pil)
        original_b64 = pil_to_base64(img_pil)

        # Explanation text
        explanation_text = (
            f"Highlighted regions indicate areas that contributed strongly to the model's {tgt_severity} prediction. "
            "Attribution represents model attention patterns rather than direct lesion etiology."
        )

        response_payload = {
            "prediction": pred_idx,
            "severity": pred_severity,
            "confidence": confidence,
            "probabilities": probs,
            "target_class": tgt_idx,
            "target_severity": tgt_severity,
            "target_layer": layer_name,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64,
            "original": original_b64,
            "sanity_stats": sanity,
            "explanation": explanation_text,
            "disclaimer": MEDICAL_DISCLAIMER
        }

        # Optional U-Net Lesion Overlap
        if with_unet:
            try:
                from inference.predict_unet import predict_retinal_lesions
                unet_res = predict_retinal_lesions(image_input=raw_bytes, img_size=512)
                # Compute overlap
                heat_512 = np.array(Image.fromarray((heatmap_raw * 255).astype(np.uint8)).resize((512, 512), Image.BICUBIC)) / 255.0
                # Binarize unet mask if available
                pred_mask_path = unet_res["predicted_mask_path"]
                # Provide unet findings
                response_payload["unet_findings"] = {
                    "detections": unet_res["detections"],
                    "pixel_coverage": unet_res["pixel_coverage_percentage"],
                    "mask": unet_res["predicted_mask_path"],
                    "overlay": unet_res["lesion_overlay_path"]
                }
            except Exception as e:
                response_payload["unet_error"] = str(e)

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": f"GRAD-CAM GENERATION FAILED: {str(e)}"}), 500


@app.route("/api/explain/shap", methods=["POST"])
@app.route("/api/shap", methods=["POST"])
def explain_shap():
    """
    POST /api/explain/shap & POST /api/shap
    Input: Retinal fundus image (file upload, raw bytes, or JSON {"image": "<base64/path>"})
           Optional params: target_class (0..4), nsamples (default 30), with_unet (bool)
    Output: JSON conforming to Section 23 specification with prediction, positive/negative
            visualizations, overlays, statistics, and clinical disclaimers.
    """
    import base64
    import io
    import numpy as np
    from PIL import Image
    from explainability.shap_explainer import generate_shap_explanation
    from explainability.shap_visualizer import (
        pil_to_base64,
        create_positive_map,
        create_negative_map,
        create_combined_map,
        create_overlay
    )
    from explainability.shap_summary import (
        generate_human_readable_summary,
        compute_lesion_aware_shap_stats,
        compute_shap_unet_overlap,
        MEDICAL_SAFETY_DISCLAIMER
    )

    raw_bytes = None
    target_class = None
    nsamples = 30
    with_unet = False

    # Check query params
    if request.args.get("target_class") is not None:
        try:
            target_class = int(request.args.get("target_class"))
        except ValueError:
            target_class = None
    if request.args.get("nsamples") is not None:
        try:
            nsamples = max(10, min(100, int(request.args.get("nsamples"))))
        except ValueError:
            nsamples = 30
    if request.args.get("with_unet") is not None:
        with_unet = request.args.get("with_unet").lower() in ["1", "true", "yes"]

    # Check JSON payload
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if "target_class" in data and target_class is None:
            try:
                target_class = int(data["target_class"])
            except (ValueError, TypeError):
                target_class = None
        if "nsamples" in data:
            try:
                nsamples = max(10, min(100, int(data["nsamples"])))
            except (ValueError, TypeError):
                pass
        if "with_unet" in data:
            with_unet = bool(data["with_unet"])

        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    # Check multipart files
    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    # Check raw data
    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    # Check GET query image parameter
    if raw_bytes is None and request.args.get("image"):
        img_param = request.args.get("image")
        if os.path.isfile(img_param):
            with open(img_param, "rb") as f:
                raw_bytes = f.read()
        elif os.path.isfile(os.path.join(ROOT_DIR, img_param)):
            with open(os.path.join(ROOT_DIR, img_param), "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({
            "error": "INPUT IMAGE NOT FOUND: Please upload a retinal fundus image."
        }), 400

    try:
        img_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        orig_w, orig_h = img_pil.size
        eng = get_engine()
        device = eng.device
        model = eng.model

        # Compute SHAP explanation
        shap_res = generate_shap_explanation(
            image=img_pil,
            model=model,
            target_class=target_class,
            nsamples=nsamples,
            device=device
        )

        pred_class = shap_res["prediction_class"]
        pred_label = shap_res["prediction_label"]
        pred_prob = shap_res["prediction_probability"]
        tgt_class = shap_res["target_class"]
        tgt_label = shap_res["target_label"]
        pos_map = shap_res["positive_map"]
        neg_map = shap_res["negative_map"]
        comb_map = shap_res["combined_map"]
        stats = shap_res["shap_stats"]

        # Generate base64 images
        pos_img = create_positive_map(pos_map, orig_size=(orig_w, orig_h))
        neg_img = create_negative_map(neg_map, orig_size=(orig_w, orig_h))
        comb_img = create_combined_map(comb_map, orig_size=(orig_w, orig_h))
        overlay_img = create_overlay(img_pil, comb_map, alpha=0.45, show_colorbar=True)

        pos_b64 = pil_to_base64(pos_img)
        neg_b64 = pil_to_base64(neg_img)
        comb_b64 = pil_to_base64(comb_img)
        overlay_b64 = pil_to_base64(overlay_img)
        orig_b64 = pil_to_base64(img_pil)

        # Optional U-Net integration
        lesion_stats = None
        overlap_info = None
        unet_payload = None
        if with_unet:
            try:
                from inference.predict_unet import predict_retinal_lesions
                unet_res = predict_retinal_lesions(image_input=raw_bytes, img_size=512)
                if "class_masks" in unet_res:
                    lesion_stats = compute_lesion_aware_shap_stats(comb_map, unet_res["class_masks"])
                if "combined_mask" in unet_res:
                    overlap_info = compute_shap_unet_overlap(comb_map, unet_res["combined_mask"])
                unet_payload = {
                    "detections": unet_res["detections"],
                    "pixel_coverage": unet_res["pixel_coverage_percentage"],
                    "overlay": unet_res["lesion_overlay_path"]
                }
            except Exception as ue:
                unet_payload = {"error": str(ue)}

        # Human-readable summary
        summary = generate_human_readable_summary(
            prediction_class=pred_class,
            prediction_label=pred_label,
            confidence=pred_prob,
            target_class=tgt_class,
            target_label=tgt_label,
            shap_stats=stats,
            lesion_stats=lesion_stats,
            overlap_info=overlap_info
        )

        response_payload = {
            "prediction": {
                "class_id": pred_class,
                "label": pred_label,
                "confidence": pred_prob,
                "probabilities": shap_res["probabilities"]
            },
            "target_class": tgt_class,
            "target_label": tgt_label,
            "explanation": {
                "method": "SHAP",
                "explainer": "shap.GradientExplainer (Expected Gradients)",
                "positive_regions": summary["positive_contribution_narrative"],
                "negative_regions": summary["negative_contribution_narrative"],
                "sparsity": summary["sparsity"],
                "formatted_text": summary["formatted_text"],
                "statistics": stats
            },
            "visualizations": {
                "positive": pos_b64,
                "negative": neg_b64,
                "combined": comb_b64,
                "overlay": overlay_b64,
                "original": orig_b64
            },
            "disclaimer": MEDICAL_SAFETY_DISCLAIMER
        }

        if lesion_stats:
            response_payload["lesion_contributions"] = lesion_stats
        if overlap_info:
            response_payload["shap_unet_overlap"] = overlap_info
        if unet_payload:
            response_payload["unet_findings"] = unet_payload

        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({
            "error": f"Unable to generate SHAP explanation. The prediction model is available, but the SHAP computation could not be completed. Details: {str(e)}"
        }), 500


@app.route("/api/iqa", methods=["POST"])
@app.route("/api/quality", methods=["POST"])
def assess_image_quality():
    """
    POST /api/iqa & POST /api/quality
    Input: Retinal fundus image (file upload, raw bytes, or JSON {"image": "..."})
           Optional query param: threshold (default 0.75)
    Output: JSON with quality class, confidence, status (ACCEPTED/REJECTED),
            screening_allowed (bool), and auxiliary indicators.
    """
    import base64
    import io
    from inference.predict_iqa import IQAInferenceEngine
    from models.iqa_model import DEFAULT_GOOD_THRESHOLD

    threshold = DEFAULT_GOOD_THRESHOLD
    if request.args.get("threshold") is not None:
        try:
            threshold = float(request.args.get("threshold"))
        except ValueError:
            threshold = DEFAULT_GOOD_THRESHOLD

    raw_bytes = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if "threshold" in data:
            try:
                threshold = float(data["threshold"])
            except (ValueError, TypeError):
                pass
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    if raw_bytes is None and request.args.get("image"):
        img_param = request.args.get("image")
        if os.path.isfile(img_param):
            with open(img_param, "rb") as f:
                raw_bytes = f.read()
        elif os.path.isfile(os.path.join(ROOT_DIR, img_param)):
            with open(os.path.join(ROOT_DIR, img_param), "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({"error": "No retinal fundus image provided."}), 400

    try:
        engine = IQAInferenceEngine(threshold=threshold)
        result = engine.predict(raw_bytes, threshold=threshold)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Image quality assessment failed: {str(e)}"}), 500


@app.route("/api/screen", methods=["POST"])
def screen_pipeline_endpoint():
    """
    POST /api/screen
    Full IQA-gated screening entrypoint.
    If image fails IQA (Poor, Unusable, or Low Confidence):
      returns status BLOCKED, dr_prediction = null, and recommendation.
    If image passes IQA (Good Quality & confident):
      runs EfficientNet-B0 and returns complete screening and explainability report.
    """
    import base64
    from inference.screening_pipeline import screen_retinal_image
    from models.iqa_model import DEFAULT_GOOD_THRESHOLD

    threshold = DEFAULT_GOOD_THRESHOLD
    if request.args.get("threshold") is not None:
        try:
            threshold = float(request.args.get("threshold"))
        except ValueError:
            threshold = DEFAULT_GOOD_THRESHOLD

    raw_bytes = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        if "threshold" in data:
            try:
                threshold = float(data["threshold"])
            except (ValueError, TypeError):
                pass
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    if raw_bytes is None and request.args.get("image"):
        img_param = request.args.get("image")
        if os.path.isfile(img_param):
            with open(img_param, "rb") as f:
                raw_bytes = f.read()
        elif os.path.isfile(os.path.join(ROOT_DIR, img_param)):
            with open(os.path.join(ROOT_DIR, img_param), "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({"error": "No retinal fundus image provided."}), 400

    try:
        result = screen_retinal_image(raw_bytes, threshold=threshold)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Screening pipeline error: {str(e)}"}), 500


@app.route("/api/referral", methods=["GET", "POST"])
def referral_endpoint():
    """
    POST /api/referral (also supports GET for testing)
    Rule-based clinical decision endpoint mapping DR grade to risk category and referral priority.
    Input parameters (JSON or query params):
      - dr_grade (int): 0-4
      - dr_confidence (float): 0.0-1.0
      - image_quality_status (str, optional): e.g. "ACCEPTED", "REJECTED"
      - image_quality_confidence (float, optional)
      - minimum_dr_confidence (float, optional, default 0.50)
      - medical_report (dict, optional): optional clinical values (glucose, hba1c, bp)
    """
    from decision.referral_engine import generate_referral_decision
    from decision.referral_summary import format_clinical_screening_summary

    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}

    # Extract dr_grade
    dr_grade_raw = data.get("dr_grade", request.args.get("dr_grade"))
    dr_grade = None
    if dr_grade_raw is not None:
        try:
            dr_grade = int(dr_grade_raw)
        except (ValueError, TypeError):
            dr_grade = dr_grade_raw

    # Extract dr_confidence
    dr_conf_raw = data.get("dr_confidence", request.args.get("dr_confidence"))
    dr_confidence = None
    if dr_conf_raw is not None:
        try:
            dr_confidence = float(dr_conf_raw)
        except (ValueError, TypeError):
            dr_confidence = dr_conf_raw

    # Extract quality status
    iq_status = data.get("image_quality_status", request.args.get("image_quality_status", "ACCEPTED"))
    iq_conf_raw = data.get("image_quality_confidence", request.args.get("image_quality_confidence", 1.0))
    try:
        iq_conf = float(iq_conf_raw) if iq_conf_raw is not None else 1.0
    except (ValueError, TypeError):
        iq_conf = 1.0

    min_conf_raw = data.get("minimum_dr_confidence", request.args.get("minimum_dr_confidence", 0.50))
    try:
        min_conf = float(min_conf_raw) if min_conf_raw is not None else 0.50
    except (ValueError, TypeError):
        min_conf = 0.50

    decision = generate_referral_decision(
        dr_grade=dr_grade,
        dr_confidence=dr_confidence,
        image_quality_status=iq_status,
        image_quality_confidence=iq_conf,
        minimum_dr_confidence=min_conf
    )

    # Optional clinical summary format
    med_data = data.get("medical_report")
    if med_data and isinstance(med_data, dict):
        summary = format_clinical_screening_summary(
            referral_decision=decision,
            medical_report_data=med_data
        )
        return jsonify({
            "referral_decision": decision,
            "clinical_summary": summary
        }), (400 if decision.get("decision_status") == "INVALID_INPUT" else 200)

    status_code = 400 if decision.get("decision_status") == "INVALID_INPUT" else 200
    return jsonify(decision), status_code


@app.route("/api/analyze-report", methods=["POST"])
def analyze_report_endpoint():
    """
    POST /api/analyze-report
    Ingests medical/laboratory report as PDF or image (JPG/PNG).
    Extracts glucose, HbA1c, blood pressure, and lab reference ranges.
    """
    import base64
    from report_analysis.analyzer import get_health_analyzer

    filename = "report_document"
    raw_bytes = None

    # Check multipart files
    if len(request.files) > 0:
        file_obj = request.files.get("report") or request.files.get("file") or next(iter(request.files.values()))
        filename = file_obj.filename or "uploaded_report"
        raw_bytes = file_obj.read()

    # Check JSON payload
    if raw_bytes is None and request.is_json:
        data = request.get_json(silent=True) or {}
        filename = data.get("filename", "report_document")
        file_val = data.get("report") or data.get("file") or data.get("image")
        if file_val:
            if isinstance(file_val, str) and file_val.startswith("data:"):
                # Data URL
                header, encoded = file_val.split(",", 1)
                file_val = encoded
                if "pdf" in header:
                    filename = filename if filename.endswith(".pdf") else f"{filename}.pdf"
                elif "png" in header:
                    filename = filename if filename.endswith(".png") else f"{filename}.png"
                elif "jpeg" in header or "jpg" in header:
                    filename = filename if filename.endswith(".jpg") else f"{filename}.jpg"
            try:
                raw_bytes = base64.b64decode(file_val)
            except Exception:
                if os.path.isfile(file_val):
                    filename = os.path.basename(file_val)
                    with open(file_val, "rb") as f:
                        raw_bytes = f.read()

    # Check raw binary request body
    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data
        filename = request.args.get("filename", "report_document")

    # Check GET/POST query path
    if raw_bytes is None and request.args.get("file"):
        f_path = request.args.get("file")
        if os.path.isfile(f_path):
            filename = os.path.basename(f_path)
            with open(f_path, "rb") as f:
                raw_bytes = f.read()
        elif os.path.isfile(os.path.join(ROOT_DIR, f_path)):
            filename = os.path.basename(f_path)
            with open(os.path.join(ROOT_DIR, f_path), "rb") as f:
                raw_bytes = f.read()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({
            "success": False,
            "error": "No medical report document provided. Please upload a PDF, JPG, JPEG, or PNG file."
        }), 400

    analyzer = get_health_analyzer()
    result = analyzer.analyze_report(raw_bytes, filename=filename)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/health-summary", methods=["POST"])
def health_summary_endpoint():
    """
    POST /api/health-summary
    Merges report-derived health measurements and manual inputs with DR screening findings.
    """
    from report_analysis.analyzer import get_health_analyzer

    data = request.get_json(silent=True) or {}
    report_data = data.get("report_data")
    manual_glucose = data.get("manual_glucose")
    manual_bp = data.get("manual_bp")
    dr_prediction = data.get("dr_prediction")
    lesions_data = data.get("lesions_data")
    referral_decision = data.get("referral_decision")

    analyzer = get_health_analyzer()
    merged_health = analyzer.merge_with_manual_entries(
        report_data=report_data,
        manual_glucose=manual_glucose,
        manual_bp=manual_bp
    )

    combined_summary = analyzer.build_combined_screening_summary(
        dr_prediction=dr_prediction,
        health_data=merged_health,
        lesions_data=lesions_data,
        referral_decision=referral_decision
    )

    return jsonify({
        "success": True,
        "health_data": merged_health,
        "combined_summary": combined_summary
    }), 200



@app.route("/api/retinal-analysis", methods=["POST"])
def retinal_analysis_endpoint():
    """
    POST /api/retinal-analysis
    Executes unified retinal anatomical structure and lesion analysis:
    - Optic Disc Localization
    - Fovea Localization
    - Retinal Vessel Segmentation
    - Lesion Analysis (Microaneurysms, Hemorrhages, Hard/Soft Exudates)
    - Neovascularization (Reported as PENDING DATASET)
    """
    import base64
    from retinal_analysis.pipeline import analyze_retinal_structures

    raw_bytes = None

    # 1. Check JSON payload
    if request.is_json:
        data = request.get_json(silent=True) or {}
        img_val = data.get("image")
        if img_val:
            if isinstance(img_val, str) and img_val.startswith("data:"):
                img_val = img_val.split(",", 1)[-1]
            try:
                raw_bytes = base64.b64decode(img_val)
            except Exception:
                if os.path.isfile(img_val):
                    with open(img_val, "rb") as f:
                        raw_bytes = f.read()

    # 2. Check request.files
    if raw_bytes is None and len(request.files) > 0:
        file_obj = request.files.get("image") or request.files.get("file") or next(iter(request.files.values()))
        raw_bytes = file_obj.read()

    # 3. Check request.data
    if raw_bytes is None and request.data and len(request.data) > 0:
        raw_bytes = request.data

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({
            "error": "No retinal fundus image provided. Send an image via file upload, raw body, or JSON {'image': 'base64'}."
        }), 400

    try:
        res = analyze_retinal_structures(raw_bytes, include_base64=True)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": f"Retinal analysis failed: {str(e)}"}), 500


# ============================================================
# PATIENT PORTAL WEB ROUTES
# ============================================================

@app.route("/patient/login", methods=["GET"])
def patient_login():
    return send_from_directory(ROOT_DIR, "patient-login.html")

@app.route("/patient/signup", methods=["GET"])
def patient_signup():
    return send_from_directory(ROOT_DIR, "patient-signup.html")

@app.route("/patient/dashboard", methods=["GET"])
def patient_dashboard():
    return send_from_directory(ROOT_DIR, "patient-dashboard.html")

@app.route("/patient/screening", methods=["GET"])
@app.route("/patient/screening/<screening_id>", methods=["GET"])
def patient_screening(screening_id=None):
    return send_from_directory(ROOT_DIR, "patient-screening.html")

@app.route("/patient/profile", methods=["GET"])
def patient_profile():
    return send_from_directory(ROOT_DIR, "patient-profile.html")


# ============================================================
# CLINICAL DOCTOR PORTAL WEB ROUTES
# ============================================================

@app.route("/doctor/login", methods=["GET"])
def doctor_login():
    return send_from_directory(ROOT_DIR, "doctor-login.html")

@app.route("/doctor/dashboard", methods=["GET"])
def doctor_dashboard():
    return send_from_directory(ROOT_DIR, "doctor-dashboard.html")

@app.route("/doctor/patient/<patient_uid>", methods=["GET"])
def doctor_patient(patient_uid):
    return send_from_directory(ROOT_DIR, "doctor-patient.html")

@app.route("/doctor/patient/<patient_uid>/screening/<screening_id>", methods=["GET"])
def doctor_screening(patient_uid, screening_id):
    return send_from_directory(ROOT_DIR, "doctor-screening.html")


# ============================================================
# CLINICAL DOCTOR DOCUMENT UPLOAD & DOWNLOAD APIS
# ============================================================

CLINICAL_UPLOAD_DIR = os.path.join(ROOT_DIR, "data", "clinical_reports")
os.makedirs(CLINICAL_UPLOAD_DIR, exist_ok=True)

@app.route("/api/doctor/upload-document", methods=["POST"])
def upload_clinical_document():
    """
    POST /api/doctor/upload-document
    Supports uploading medical clinical reports and supporting documents (PDF, JPG, PNG)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Form field 'file' is required."}), 400
    
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected."}), 400
        
    allowed_exts = {".pdf", ".jpg", ".jpeg", ".png"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_exts:
        return jsonify({"error": f"Invalid file type '{ext}'. Only PDF, JPG, and PNG are supported."}), 400

    patient_uid = request.form.get("patientUid", "usr_unknown")
    screening_id = request.form.get("screeningId", "SCR_UNKNOWN")
    title = request.form.get("title", f.filename)
    description = request.form.get("description", "Doctor Clinical Evaluation Report")
    doctor_name = request.form.get("doctorName", "Clinical Doctor")

    import uuid
    import datetime
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    safe_name = f"{doc_id}_{f.filename.replace(' ', '_')}"
    
    target_dir = os.path.join(CLINICAL_UPLOAD_DIR, patient_uid, screening_id)
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, safe_name)
    f.save(file_path)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    file_size_kb = f"{round(os.path.getsize(file_path) / 1024)} KB"
    download_url = f"/api/doctor/document/{patient_uid}/{screening_id}/{safe_name}"

    return jsonify({
        "status": "success",
        "document": {
            "documentId": doc_id,
            "fileName": f.filename,
            "safeName": safe_name,
            "title": title,
            "description": description,
            "fileType": f.content_type or ("application/pdf" if ext == ".pdf" else "image/jpeg"),
            "fileSize": file_size_kb,
            "uploadedBy": doctor_name,
            "uploadedAt": now_iso,
            "relatedScreening": screening_id,
            "patientUid": patient_uid,
            "url": download_url
        }
    }), 201

@app.route("/api/doctor/document/<patient_uid>/<screening_id>/<filename>", methods=["GET"])
def download_clinical_document(patient_uid, screening_id, filename):
    safe_dir = os.path.join(CLINICAL_UPLOAD_DIR, patient_uid, screening_id)
    if not os.path.exists(os.path.join(safe_dir, filename)):
        return jsonify({"error": "Document not found."}), 404
    return send_from_directory(safe_dir, filename)


# Serve static frontend files


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return ("", 204)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(ROOT_DIR, "index.html")

@app.route("/settings", methods=["GET"])
def settings_route():
    return send_from_directory(ROOT_DIR, "settings.html")

@app.route("/<path:path>", methods=["GET"])
def serve_file(path):
    if os.path.exists(os.path.join(ROOT_DIR, path)):
        return send_from_directory(ROOT_DIR, path)
    return jsonify({"error": f"File '{path}' not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting DR Screening AI Backend on port {port}...")
    print(f"Hardware device: {get_device()}")
    app.run(host="0.0.0.0", port=port, debug=False)
