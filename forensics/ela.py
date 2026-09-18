import concurrent.futures
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import base64


def error_level_analysis(image_file, quality=90):
    """
    Performs Error Level Analysis (ELA) to detect tampering or compression artifacts.

    NOTE: ELA is a splicing/tampering detector, not an AI-generation detector.
    Its anomaly score is combined with frequency_domain_analysis() for a
    composite forensic signal.
    """
    try:
        image_file.seek(0)
        original = Image.open(image_file)
        is_jpeg = original.format in ('JPEG', 'MPO')
        original = original.convert('RGB')

        # Save at given quality
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, 'JPEG', quality=quality)
        temp_buffer.seek(0)
        compressed = Image.open(temp_buffer)

        # Calculate difference
        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()

        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Save ELA image to bytes for display
        ela_buffer = io.BytesIO()
        ela_image.save(ela_buffer, format="JPEG")
        ela_base64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')

        # Generate a signal score based on the variance of the ELA image
        ela_gray = ela_image.convert('L')
        ela_array = np.array(ela_gray)
        variance = np.var(ela_array)

        # Local contrast/anomaly heuristic instead of raw variance
        median_val = np.median(ela_array)
        p95_val = np.percentile(ela_array, 95)
        local_anomaly = float(p95_val - median_val) / 255.0

        # Calculate anomaly score (0 to 1)
        anomaly_score = min(1.0, local_anomaly * 1.5)

        return {
            "variance": float(variance),
            "anomaly_score": float(anomaly_score),
            "ela_max_diff": float(max_diff),
            "ela_image_base64": ela_base64,
            "ela_reliable": is_jpeg
        }

    except Exception as e:
        return {"error": str(e)}


def frequency_domain_analysis(image_file):
    """
    Analyzes the frequency spectrum of an image to detect AI-generation artifacts.

    AI-generated images typically exhibit:
    - Suppressed high-frequency components (missing camera sensor noise)
    - Periodic spectral peaks from generator architectures (GAN grid artifacts)
    - Unnaturally smooth DCT coefficient distributions

    Returns a dict with:
    - hf_ratio: high-freq to total energy ratio. Lower = more suspicious of AI.
    - spectral_entropy: entropy of the magnitude spectrum. Lower = more suspicious.
    - ai_frequency_score: composite 0–1 score (higher = more likely AI).
    - spectrum_image_base64: visualisation of the magnitude spectrum.
    """
    try:
        image_file.seek(0)
        img = Image.open(image_file).convert('L')
        # Resize to fixed resolution so spectral entropy bounds are consistent across all images
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # 2D FFT → shift zero-frequency to centre
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1.0)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Define "high frequency" as the outer 50% ring of the spectrum
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)

        low_freq_mask = dist_from_center <= (max_dist * 0.25)
        high_freq_mask = dist_from_center > (max_dist * 0.5)

        low_freq_energy = float(magnitude[low_freq_mask].mean()) if low_freq_mask.any() else 0.0
        high_freq_energy = float(magnitude[high_freq_mask].mean()) if high_freq_mask.any() else 0.0
        total_energy = float(magnitude.mean())

        # HF ratio: real photos have more high-frequency content (sensor noise)
        hf_ratio = high_freq_energy / total_energy if total_energy > 0 else 0.0

        # Spectral entropy: AI images tend to have lower spectral entropy
        mag_norm = magnitude / magnitude.sum() if magnitude.sum() > 0 else magnitude
        mag_flat = mag_norm.flatten()
        mag_flat = mag_flat[mag_flat > 0]  # avoid log(0)
        spectral_entropy = float(-np.sum(mag_flat * np.log2(mag_flat)))

        # Normalise spectral entropy to 0–1 range (empirical bounds)
        # For 512x512, max theoretical entropy is log2(262144) = 18.0
        # Adjusted bounds for 512x512: real photos ~15–17; AI images ~11–14
        entropy_norm = np.clip((spectral_entropy - 10.0) / 7.0, 0.0, 1.0)

        # Composite AI frequency score (higher = more likely AI)
        # Low HF ratio + low entropy → suspicious
        hf_score = 1.0 - np.clip(hf_ratio / 0.8, 0.0, 1.0)  # normalise; real photos ~0.5–0.8
        entropy_score = 1.0 - entropy_norm
        ai_frequency_score = float(np.clip(0.6 * hf_score + 0.4 * entropy_score, 0.0, 1.0))

        # Generate spectrum visualisation
        mag_vis = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255).astype(np.uint8)
        spectrum_img = Image.fromarray(mag_vis)
        spec_buffer = io.BytesIO()
        spectrum_img.save(spec_buffer, format="JPEG")
        spectrum_base64 = base64.b64encode(spec_buffer.getvalue()).decode('utf-8')

        return {
            "hf_ratio": float(hf_ratio),
            "spectral_entropy": float(spectral_entropy),
            "ai_frequency_score": float(ai_frequency_score),
            "spectrum_image_base64": spectrum_base64,
        }

    except Exception as e:
        return {"error": str(e)}


def combined_forensic_analysis(image_file, ela_quality=90):
    """
    Runs both ELA and frequency-domain analysis and produces a composite
    forensic anomaly score.

    The composite score down-weights ELA (which detects compression/tampering,
    not AI generation) and up-weights frequency-domain signals.
    """
    image_file.seek(0)
    file_bytes = image_file.read()

    # Run ELA and frequency analysis concurrently — they are fully independent
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_ela = pool.submit(error_level_analysis, io.BytesIO(file_bytes), ela_quality)
        f_freq = pool.submit(frequency_domain_analysis, io.BytesIO(file_bytes))
        ela_result = f_ela.result()
        freq_result = f_freq.result()

    # Extract individual scores (defaulting to neutral on error)
    ela_score = ela_result.get("anomaly_score", 0.0) if "error" not in ela_result else 0.0
    
    # Down-weight ELA if the original image wasn't a JPEG (first-generation quantization loss)
    if "error" not in ela_result and not ela_result.get("ela_reliable", True):
        ela_score *= 0.2

    freq_score = freq_result.get("ai_frequency_score", 0.0) if "error" not in freq_result else 0.0

    # Composite: frequency analysis is the primary AI signal, ELA is supplementary
    composite = 0.3 * ela_score + 0.7 * freq_score

    result = {
        "anomaly_score": float(composite),
        "ela": ela_result,
        "frequency": freq_result,
    }

    # Propagate visualisations to top level for backward compatibility
    if "ela_image_base64" in ela_result:
        result["ela_image_base64"] = ela_result["ela_image_base64"]
    if "spectrum_image_base64" in freq_result:
        result["spectrum_image_base64"] = freq_result["spectrum_image_base64"]
    # Propagate sub-scores for display
    result["variance"] = ela_result.get("variance", 0.0)
    result["ela_max_diff"] = ela_result.get("ela_max_diff", 0.0)

    return result
