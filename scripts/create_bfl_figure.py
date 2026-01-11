"""Create BFL comparison figure with torchmetrics - full + zoom metrics.

Usage: modal run scripts/create_bfl_figure.py
"""
import modal

app = modal.App("bfl-figure")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("fonts-dejavu-core")
    .pip_install("torch", "torchmetrics", "Pillow", "numpy")
)

data_volume = modal.Volume.from_name("vitok-data")


@app.function(image=image, gpu="T4", timeout=300, volumes={"/data": data_volume})
def create_figure():
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from pathlib import Path

    print("Initializing torchmetrics...")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()

    def compute_metrics(img1, img2):
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        return psnr_metric(t1, t2).item(), ssim_metric(t1, t2).item()

    # Paths - full images
    orig_dir = Path("/data/bfl_compare/bfl_upload/originals")
    vitok_dir = Path("/data/bfl_compare/bfl_upload/vitok")
    flux2_dir = Path("/data/bfl_compare/bfl_upload/flux2")

    # Paths - zoom crops
    orig_zoom_dir = Path("/data/bfl_compare/zooms/bfl_upload_zooms/orig_zoom")
    vitok_zoom_dir = Path("/data/bfl_compare/zooms/bfl_upload_zooms/vitok_zoom")
    flux2_zoom_dir = Path("/data/bfl_compare/zooms/bfl_upload_zooms/flux2_zoom")

    output_dir = Path("/data/bfl_compare/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_metrics = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        print("Loaded DejaVu fonts")
    except:
        font_title = font_header = font_metrics = font_small = ImageFont.load_default()
        print("Using default font")

    # Collect data for all scenes
    scenes_data = []
    for scene in ["scene1", "scene2", "scene3"]:
        # Load full images
        orig_np = np.array(Image.open(orig_dir / f"{scene}.png").convert("RGB"))
        vitok_np = np.array(Image.open(vitok_dir / f"{scene}.png").convert("RGB"))
        flux2_np = np.array(Image.open(flux2_dir / f"{scene}_flux2.png").convert("RGB"))

        min_h = min(orig_np.shape[0], vitok_np.shape[0], flux2_np.shape[0])
        min_w = min(orig_np.shape[1], vitok_np.shape[1], flux2_np.shape[1])
        orig_np = orig_np[:min_h, :min_w]
        vitok_np = vitok_np[:min_h, :min_w]
        flux2_np = flux2_np[:min_h, :min_w]

        # Load zoom crops
        orig_zoom_np = np.array(Image.open(orig_zoom_dir / f"{scene}.png").convert("RGB"))
        vitok_zoom_np = np.array(Image.open(vitok_zoom_dir / f"{scene}_vitok.png").convert("RGB"))
        flux2_zoom_np = np.array(Image.open(flux2_zoom_dir / f"{scene}.png").convert("RGB"))

        # Compute FULL image metrics
        full_vitok_psnr, full_vitok_ssim = compute_metrics(orig_np, vitok_np)
        full_flux2_psnr, full_flux2_ssim = compute_metrics(orig_np, flux2_np)

        # Compute ZOOM metrics
        zoom_vitok_psnr, zoom_vitok_ssim = compute_metrics(orig_zoom_np, vitok_zoom_np)
        zoom_flux2_psnr, zoom_flux2_ssim = compute_metrics(orig_zoom_np, flux2_zoom_np)

        print(f"\n{scene}:")
        print(f"  FULL  - ViTok: {full_vitok_psnr:.1f}dB/{full_vitok_ssim:.3f} | FLUX.2: {full_flux2_psnr:.1f}dB/{full_flux2_ssim:.3f}")
        print(f"  ZOOM  - ViTok: {zoom_vitok_psnr:.1f}dB/{zoom_vitok_ssim:.3f} | FLUX.2: {zoom_flux2_psnr:.1f}dB/{zoom_flux2_ssim:.3f}")

        scenes_data.append({
            "name": scene,
            "orig": Image.fromarray(orig_np),
            "vitok": Image.fromarray(vitok_np),
            "flux2": Image.fromarray(flux2_np),
            "orig_zoom": Image.fromarray(orig_zoom_np),
            "vitok_zoom": Image.fromarray(vitok_zoom_np),
            "flux2_zoom": Image.fromarray(flux2_zoom_np),
            "full_vitok_psnr": full_vitok_psnr, "full_vitok_ssim": full_vitok_ssim,
            "full_flux2_psnr": full_flux2_psnr, "full_flux2_ssim": full_flux2_ssim,
            "zoom_vitok_psnr": zoom_vitok_psnr, "zoom_vitok_ssim": zoom_vitok_ssim,
            "zoom_flux2_psnr": zoom_flux2_psnr, "zoom_flux2_ssim": zoom_flux2_ssim,
        })

    # Create figure - layout: [Full Orig | Full ViTok | Full FLUX.2 | Zoom Orig | Zoom ViTok | Zoom FLUX.2]
    print("\nCreating figure...")

    full_w, full_h = scenes_data[0]["orig"].size
    zoom_w, zoom_h = scenes_data[0]["orig_zoom"].size

    # Scale zoom to match full height
    zoom_scale = full_h / zoom_h
    scaled_zoom_w = int(zoom_w * zoom_scale)
    scaled_zoom_h = full_h

    margin = 20
    section_gap = 40  # Gap between full and zoom sections
    label_height = 90
    title_height = 70
    padding = 30

    # Total width: 3 full images + gap + 3 scaled zoom images
    fig_width = (full_w * 3 + margin * 2) + section_gap + (scaled_zoom_w * 3 + margin * 2) + padding * 2
    fig_height = title_height + (full_h + label_height) * 3 + margin * 2 + padding

    fig = Image.new("RGB", (fig_width, fig_height), (20, 22, 28))
    draw = ImageDraw.Draw(fig)

    # Title
    title = "ViTok L-64 vs FLUX.2 VAE Reconstruction"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    draw.text(((fig_width - (bbox[2]-bbox[0])) // 2, 18), title, fill=(255, 255, 255), font=font_title)

    # Section headers
    full_section_x = padding + (full_w * 3 + margin * 2) // 2
    zoom_section_x = padding + (full_w * 3 + margin * 2) + section_gap + (scaled_zoom_w * 3 + margin * 2) // 2

    draw.text((full_section_x - 80, title_height - 25), "Full Image", fill=(180, 180, 180), font=font_header)
    draw.text((zoom_section_x - 100, title_height - 25), "Zoomed Region", fill=(180, 180, 180), font=font_header)

    # Add scenes
    y_start = title_height + 15
    for idx, data in enumerate(scenes_data):
        y = y_start + idx * (full_h + label_height + margin)

        # === FULL IMAGES ===
        x_full = padding
        fig.paste(data["orig"], (x_full, y))
        fig.paste(data["vitok"], (x_full + full_w + margin, y))
        fig.paste(data["flux2"], (x_full + full_w * 2 + margin * 2, y))

        # Full image labels (under ViTok and FLUX.2)
        full_vitok_better = data["full_vitok_psnr"] > data["full_flux2_psnr"]

        # ViTok full metrics
        vitok_text = f"{data['full_vitok_psnr']:.1f}dB | {data['full_vitok_ssim']:.3f}"
        vitok_color = (80, 200, 80) if full_vitok_better else (200, 80, 80)
        x_vitok = x_full + full_w + margin + full_w // 2
        bbox = draw.textbbox((0, 0), vitok_text, font=font_metrics)
        draw.text((x_vitok - (bbox[2]-bbox[0])//2, y + full_h + 8), vitok_text, fill=vitok_color, font=font_metrics)

        # FLUX.2 full metrics
        flux2_text = f"{data['full_flux2_psnr']:.1f}dB | {data['full_flux2_ssim']:.3f}"
        flux2_color = (200, 80, 80) if full_vitok_better else (80, 200, 80)
        x_flux2 = x_full + full_w * 2 + margin * 2 + full_w // 2
        bbox = draw.textbbox((0, 0), flux2_text, font=font_metrics)
        draw.text((x_flux2 - (bbox[2]-bbox[0])//2, y + full_h + 8), flux2_text, fill=flux2_color, font=font_metrics)

        # Column labels (first row only)
        if idx == 0:
            for i, label in enumerate(["Original", "ViTok L-64", "FLUX.2"]):
                lx = x_full + i * (full_w + margin) + full_w // 2
                bbox = draw.textbbox((0, 0), label, font=font_small)
                draw.text((lx - (bbox[2]-bbox[0])//2, y + full_h + 45), label, fill=(150, 150, 150), font=font_small)

        # === ZOOM IMAGES ===
        x_zoom = padding + (full_w * 3 + margin * 2) + section_gap

        # Scale zoom images
        orig_zoom_scaled = data["orig_zoom"].resize((scaled_zoom_w, scaled_zoom_h), Image.Resampling.LANCZOS)
        vitok_zoom_scaled = data["vitok_zoom"].resize((scaled_zoom_w, scaled_zoom_h), Image.Resampling.LANCZOS)
        flux2_zoom_scaled = data["flux2_zoom"].resize((scaled_zoom_w, scaled_zoom_h), Image.Resampling.LANCZOS)

        fig.paste(orig_zoom_scaled, (x_zoom, y))
        fig.paste(vitok_zoom_scaled, (x_zoom + scaled_zoom_w + margin, y))
        fig.paste(flux2_zoom_scaled, (x_zoom + scaled_zoom_w * 2 + margin * 2, y))

        # Zoom metrics
        zoom_vitok_better = data["zoom_vitok_psnr"] > data["zoom_flux2_psnr"]

        # ViTok zoom metrics
        vitok_zoom_text = f"{data['zoom_vitok_psnr']:.1f}dB | {data['zoom_vitok_ssim']:.3f}"
        vitok_zoom_color = (80, 200, 80) if zoom_vitok_better else (200, 80, 80)
        x_vitok_z = x_zoom + scaled_zoom_w + margin + scaled_zoom_w // 2
        bbox = draw.textbbox((0, 0), vitok_zoom_text, font=font_metrics)
        draw.text((x_vitok_z - (bbox[2]-bbox[0])//2, y + full_h + 8), vitok_zoom_text, fill=vitok_zoom_color, font=font_metrics)

        # FLUX.2 zoom metrics
        flux2_zoom_text = f"{data['zoom_flux2_psnr']:.1f}dB | {data['zoom_flux2_ssim']:.3f}"
        flux2_zoom_color = (200, 80, 80) if zoom_vitok_better else (80, 200, 80)
        x_flux2_z = x_zoom + scaled_zoom_w * 2 + margin * 2 + scaled_zoom_w // 2
        bbox = draw.textbbox((0, 0), flux2_zoom_text, font=font_metrics)
        draw.text((x_flux2_z - (bbox[2]-bbox[0])//2, y + full_h + 8), flux2_zoom_text, fill=flux2_zoom_color, font=font_metrics)

        # Column labels for zoom (first row only)
        if idx == 0:
            for i, label in enumerate(["Original", "ViTok L-64", "FLUX.2"]):
                lx = x_zoom + i * (scaled_zoom_w + margin) + scaled_zoom_w // 2
                bbox = draw.textbbox((0, 0), label, font=font_small)
                draw.text((lx - (bbox[2]-bbox[0])//2, y + full_h + 45), label, fill=(150, 150, 150), font=font_small)

    # Save
    output_path = output_dir / "comparison_full_and_zoom.png"
    fig.save(output_path, quality=95)
    print(f"\nSaved: {output_path} ({fig_width}x{fig_height})")

    data_volume = modal.Volume.from_name("vitok-data")
    data_volume.commit()

    return {"output": str(output_path), "size": f"{fig_width}x{fig_height}"}


@app.local_entrypoint()
def main():
    result = create_figure.remote()
    print(f"\nResult: {result}")
