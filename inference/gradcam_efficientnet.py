"""
Grad-CAM Explainable AI Inference for EfficientNet-B0
Diabetic Retinopathy Screening in Rural India

Usage:
    python inference/gradcam_efficientnet.py --image path/to/fundus.jpg
    python inference/gradcam_efficientnet.py --folder path/to/images
    python inference/gradcam_efficientnet.py --image path/to/fundus.jpg --target-class 2 --alpha 0.4
    python inference/gradcam_efficientnet.py --image path/to/fundus.jpg --with-unet
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import (
    EfficientNetDRClassifier,
    build_model,
    get_device,
    CLASS_NAMES,
    CLASS_KEYS
)
from backend.model.inference import get_inference_transforms
from inference.gradcam import (
    GradCAM,
    apply_colormap_and_overlay,
    calculate_gradcam_unet_overlap,
    MEDICAL_DISCLAIMER
)


def find_model_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds existing EfficientNet-B0 weights, checking canonical paths."""
    if custom_path:
        return os.path.abspath(custom_path) if os.path.isfile(custom_path) else None

    candidates = [
        os.path.join(ROOT_DIR, "models", "efficientnet_b0_dr.pth"),
        os.path.join(ROOT_DIR, "backend", "model", "model_weights.pth"),
        os.path.join(ROOT_DIR, "training", "checkpoints", "best_model.pth")
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def run_gradcam_single_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    target_class: Optional[int] = None,
    alpha: float = 0.4,
    colormap: str = "jet",
    output_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    with_unet: bool = False,
    unet_model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes Grad-CAM explainability pipeline on a single retinal fundus photograph.
    """
    if not os.path.isfile(image_path):
        print(f"INPUT IMAGE NOT FOUND: {image_path}")
        return {"status": "error", "error": f"INPUT IMAGE NOT FOUND: {image_path}"}

    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Preprocessing - exact match with training & backend inference
    transform = get_inference_transforms()
    try:
        orig_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"INPUT IMAGE NOT FOUND or corrupted: {e}")
        return {"status": "error", "error": str(e)}

    orig_w, orig_h = orig_pil.size
    input_tensor = transform(orig_pil).unsqueeze(0).to(device)

    # 2. Get target layer
    target_layer = model.get_gradcam_target_layer()
    layer_name = "backbone.features[8][0] (Conv2d: 320 -> 1280)"

    # 3. Compute Grad-CAM
    try:
        with GradCAM(model, target_layer) as gcam:
            cam_result = gcam.generate_cam(input_tensor, target_class=target_class)
    except Exception as e:
        print(f"GRAD-CAM GENERATION FAILED: {e}")
        return {"status": "error", "error": f"GRAD-CAM GENERATION FAILED: {e}"}

    heatmap_raw = cam_result["heatmap"]
    pred_idx = cam_result["prediction"]
    pred_severity = cam_result["predicted_severity"]
    confidence = cam_result["confidence"]
    probs = cam_result["probabilities"]
    tgt_idx = cam_result["target_class"]
    tgt_severity = cam_result["target_severity"]
    sanity = cam_result["sanity_stats"]

    # 4. Colorize and create overlay
    orig_np, heatmap_rgb, heatmap_pil, overlay_pil = apply_colormap_and_overlay(
        original_img=orig_pil,
        heatmap=heatmap_raw,
        alpha=alpha,
        colormap_name=colormap
    )

    # 5. Determine output filenames
    if filename_prefix:
        orig_filename = f"{filename_prefix}_original.png"
        heat_filename = f"{filename_prefix}_heatmap.png"
        over_filename = f"{filename_prefix}_overlay.png"
        npy_filename = f"{filename_prefix}_heatmap.npy"
    else:
        orig_filename = "original.png"
        heat_filename = "heatmap.png"
        over_filename = "overlay.png"
        npy_filename = "heatmap.npy"

    orig_path = os.path.join(output_dir, orig_filename)
    heat_path = os.path.join(output_dir, heat_filename)
    over_path = os.path.join(output_dir, over_filename)
    npy_path = os.path.join(output_dir, npy_filename)

    orig_pil.save(orig_path)
    heatmap_pil.save(heat_path)
    overlay_pil.save(over_path)
    np.save(npy_path, heatmap_raw)

    # 6. Optional U-Net Lesion Integration & Quantitative Overlap
    unet_results = None
    overlap_results = None
    combined_vis_path = None

    if with_unet:
        try:
            from inference.predict_unet import predict_retinal_lesions
            unet_out_dir = os.path.join(output_dir, "unet_integration")
            os.makedirs(unet_out_dir, exist_ok=True)
            unet_res = predict_retinal_lesions(
                image_input=image_path,
                model_path=unet_model_path,
                img_size=512,
                output_dir=unet_out_dir,
                device=device
            )
            unet_results = unet_res

            # Load generated U-Net mask for quantitative overlap analysis
            # Compute overlap with 512x512 heatmap
            mask_pil = Image.open(unet_res["predicted_mask_path"])
            # Reconstruct mask array
            # If indexed mask or color mask, read unet prediction
            from models.unet_model import build_unet
            unet_model = build_unet(weights_path=unet_model_path, device=device)
            unet_model.eval()
            with torch.no_grad():
                import torchvision.transforms.functional as TF
                from training.dataset_unet import IMAGE_NET_MEAN, IMAGE_NET_STD
                u_img = orig_pil.resize((512, 512), Image.BILINEAR)
                u_tensor = TF.to_tensor(u_img)
                u_tensor = TF.normalize(u_tensor, mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD).unsqueeze(0).to(device)
                u_logits = unet_model(u_tensor)
                pred_mask = np.argmax(F.softmax(u_logits, dim=1).squeeze(0).cpu().numpy(), axis=0)

            # Resize Grad-CAM heatmap to 512x512
            heat_512 = np.array(Image.fromarray((heatmap_raw * 255).astype(np.uint8)).resize((512, 512), Image.BICUBIC)) / 255.0
            overlap_results = calculate_gradcam_unet_overlap(heat_512, pred_mask, threshold=0.5)

            # Create 5-panel combined visualization
            combined_vis_path = os.path.join(output_dir, f"{filename_prefix or 'fundus'}_combined_explanation.png")
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.patch.set_facecolor('#0a0f1d')

            panels = [
                (orig_pil, "Original Fundus Image\n(Retinal Scan)"),
                (heatmap_pil, "Grad-CAM Heatmap\n(Model Importance)"),
                (overlay_pil, f"Grad-CAM Overlay\n({tgt_severity})"),
                (Image.open(unet_res["lesion_overlay_path"]), "U-Net Lesion Segmentation\n(Predicted Lesions)"),
                (overlay_pil, "Combined Explanation\n(Attention & Lesions)")
            ]

            for ax, (img_item, title) in zip(axes, panels):
                ax.imshow(img_item)
                ax.set_title(title, color="#e2e8f0", fontsize=12, pad=8, weight='bold')
                ax.axis("off")

            fig.text(0.5, 0.02,
                     f"Grad-CAM: Highlights regions contributing to {tgt_severity} prediction | "
                     f"U-Net: Independently segments retinal lesion clusters | {MEDICAL_DISCLAIMER[:95]}...",
                     ha="center", color="#94a3b8", fontsize=10)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(combined_vis_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: U-Net integration encountered error: {e}")

    # 7. Print Required Format
    img_basename = os.path.basename(image_path)
    print("=" * 40)
    print("GRAD-CAM EXPLAINABLE AI")
    print("=" * 40)
    print("\nImage:")
    print(img_basename)
    print("\nPrediction:")
    print(pred_severity)
    print("\nConfidence:")
    print(f"{confidence * 100:.1f}%")
    print("\nClass probabilities:")
    print(f"No DR: {probs['no_dr'] * 100:.1f}%")
    print(f"Mild DR: {probs['mild'] * 100:.1f}%")
    print(f"Moderate DR: {probs['moderate'] * 100:.1f}%")
    print(f"Severe DR: {probs['severe'] * 100:.1f}%")
    print(f"Proliferative DR: {probs['proliferative'] * 100:.1f}%")
    print("\nTarget class:")
    print(tgt_severity)
    print("\nGrad-CAM target layer:")
    print(layer_name)
    print("\nHeatmap min:")
    print(sanity["min"])
    print("Heatmap max:")
    print(sanity["max"])
    print("Heatmap mean:")
    print(sanity["mean"])
    print("Heatmap dimensions:")
    print(sanity["dimensions"])
    print("\nHeatmap:")
    print(heat_path)
    print("\nOverlay:")
    print(over_path)

    if with_unet and overlap_results:
        print("\nGrad-CAM / U-Net Quantitative Overlap:")
        print(f"  Overall IoU: {overlap_results['overall_iou']}")
        print(f"  Overall Lesion Coverage: {overlap_results['overall_lesion_coverage_pct']}%")
        for l_name, l_stat in overlap_results["per_lesion"].items():
            if l_stat["detected"]:
                print(f"  {l_name.title()}: {l_stat['lesion_coverage_pct']}% covered (IoU: {l_stat['iou']})")
        if combined_vis_path:
            print(f"\nCombined Explanation saved to: {combined_vis_path}")

    print("=" * 40)

    # Explanation Text adhering strictly to requirements
    explanation_text = (
        f"Grad-CAM highlights retinal regions that contributed strongly to the {tgt_severity} classification. "
        "Highlighted regions indicate areas of high model attribution."
    )
    if with_unet and unet_results and overlap_results:
        detected = [k for k, v in unet_results["detections"].items() if v]
        if detected and overlap_results["overall_intersection"] > 0:
            explanation_text += (
                f" U-Net lesion segmentation independently identified: {', '.join(detected)}. "
                "The classifier focused on regions of the retina that overlap with or are near some predicted lesion regions."
            )

    return {
        "status": "success",
        "image": img_basename,
        "prediction": pred_idx,
        "predicted_severity": pred_severity,
        "confidence": confidence,
        "probabilities": probs,
        "target_class": tgt_idx,
        "target_severity": tgt_severity,
        "target_layer": layer_name,
        "heatmap_path": heat_path,
        "overlay_path": over_path,
        "npy_path": npy_path,
        "combined_path": combined_vis_path,
        "sanity_stats": sanity,
        "explanation": explanation_text,
        "overlap_metrics": overlap_results,
        "disclaimer": MEDICAL_DISCLAIMER
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Explainable AI Visualizations for EfficientNet-B0")
    parser.add_argument("--image", type=str, default=None, help="Path to a single retinal fundus image")
    parser.add_argument("--folder", type=str, default=None, help="Directory of retinal images for batch processing")
    parser.add_argument("--model", type=str, default=None, help="Path to trained EfficientNet-B0 checkpoint")
    parser.add_argument("--target-class", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Target class to explain (0..4). Defaults to model prediction.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha overlay transparency (0.0 to 1.0, default 0.4)")
    parser.add_argument("--colormap", type=str, default="jet", help="Colormap name ('jet', 'turbo', 'inferno')")
    parser.add_argument("--output", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--with-unet", action="store_true", default=False, help="Integrate complementary U-Net lesion segmentation")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process in batch mode")
    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Please specify either --image <path> or --folder <path>")
        sys.exit(1)

    # 1. Locate model checkpoint
    checkpoint_path = find_model_checkpoint(args.model)
    if checkpoint_path is None:
        print("EFFICIENTNET MODEL NOT FOUND")
        sys.exit(1)

    device = get_device()

    # 2. Build model with real weights
    try:
        model = build_model(num_classes=5, pretrained=False, weights_path=checkpoint_path, device=device)
        model.eval()
    except Exception as e:
        print(f"EFFICIENTNET MODEL NOT FOUND: Failed to load checkpoint {checkpoint_path}: {e}")
        sys.exit(1)

    # 3. Single Image Mode
    if args.image:
        res = run_gradcam_single_image(
            image_path=args.image,
            model=model,
            device=device,
            target_class=args.target_class,
            alpha=args.alpha,
            colormap=args.colormap,
            output_dir=args.output,
            with_unet=args.with_unet
        )
        if res.get("status") != "success":
            sys.exit(1)

    # 4. Batch Folder Mode
    elif args.folder:
        if not os.path.isdir(args.folder):
            print(f"INPUT IMAGE NOT FOUND: Directory '{args.folder}' does not exist.")
            sys.exit(1)

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(args.folder, ext)))

        if not image_files:
            print(f"INPUT IMAGE NOT FOUND: No valid images found in directory '{args.folder}'.")
            sys.exit(1)

        if args.max_images:
            image_files = image_files[:args.max_images]

        print(f"Processing {len(image_files)} images in batch mode from '{args.folder}'...")
        out_dir = args.output or os.path.join(ROOT_DIR, "results", "gradcam")

        success_count = 0
        for i, img_p in enumerate(image_files, 1):
            base = os.path.splitext(os.path.basename(img_p))[0]
            prefix = f"image_{i:03d}_{base}"
            res = run_gradcam_single_image(
                image_path=img_p,
                model=model,
                device=device,
                target_class=args.target_class,
                alpha=args.alpha,
                colormap=args.colormap,
                output_dir=out_dir,
                filename_prefix=prefix,
                with_unet=args.with_unet
            )
            if res.get("status") == "success":
                success_count += 1

        print(f"\nBatch processing finished: {success_count}/{len(image_files)} explanations generated successfully.")


if __name__ == "__main__":
    main()
