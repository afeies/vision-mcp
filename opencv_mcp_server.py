import os
import time
import cv2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("opencv-vision")

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def validate_image(image_path: str, tool: str):
    """Validate that the image exists and is a supported format. Returns (img, error_response or None)."""
    if not os.path.isfile(image_path):
        return None, error_response(tool, "FILE_NOT_FOUND",
            f"File not found: {image_path}",
            "Check the path or ask the user for the correct filename")
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return None, error_response(tool, "UNSUPPORTED_FORMAT",
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
            "Ask the user for an image in a supported format like .jpg or .png")
    img = cv2.imread(image_path)
    if img is None:
        return None, error_response(tool, "CORRUPT_IMAGE",
            f"Could not read image at {image_path}",
            "The file may be corrupt. Ask the user to provide a different image")
    return img, None


def success_response(tool: str, result: dict, suggestions: list[str] = []) -> dict:
    """Wrap a tool result in the standard response envelope."""
    return {
        "status": "success",
        "tool": tool,
        "result": result,
        "suggestions": suggestions,
    }


def error_response(tool: str, error_code: str, message: str, recovery_hint: str) -> dict:
    """Wrap an error in the standard response envelope."""
    return {
        "status": "error",
        "tool": tool,
        "error_code": error_code,
        "message": message,
        "recovery_hint": recovery_hint,
    }


def output_path(image_path: str, suffix: str) -> str:
    """Generate a unique output path using a timestamp."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    ts = int(time.time())
    return os.path.join("output", f"{base}_{suffix}_{ts}.jpg")


@mcp.tool()
def detect_edges(image_path: str, low_threshold: int = 50, high_threshold: int = 150) -> str:
    """Detect edges in an image using Canny edge detection.

    Args:
        image_path: Path to the input image.
        low_threshold: Lower threshold for Canny (default 50).
        high_threshold: Upper threshold for Canny (default 150).

    Returns:
        Path to the output edge-detected image.
    """
    img, err = validate_image(image_path, "detect_edges")
    if err:
        return err

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    out = output_path(image_path, "edges")
    cv2.imwrite(out, edges)
    return success_response("detect_edges", {
        "output_path": out,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }, suggestions=[
        "Run analyze_image on the original to compare brightness and color stats",
        "Try different thresholds — lower values detect faint edges, higher values only strong edges",
    ])


@mcp.tool()
def analyze_image(image_path: str) -> dict:
    """Analyze an image and return statistics.

    Args:
        image_path: Path to the input image.

    Returns:
        Dictionary with width, height, mean color per channel (BGR), and brightness.
    """
    img, err = validate_image(image_path, "analyze_image")
    if err:
        return err

    height, width = img.shape[:2]
    mean_bgr = img.mean(axis=(0, 1)).tolist()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    return success_response("analyze_image", {
        "width": width,
        "height": height,
        "mean_blue": round(mean_bgr[0], 2),
        "mean_green": round(mean_bgr[1], 2),
        "mean_red": round(mean_bgr[2], 2),
        "brightness": round(brightness, 2),
    }, suggestions=[
        "Run detect_edges to see the shape boundaries in this image",
        "Run detect_faces to check for people in this image",
    ])


@mcp.tool()
def detect_faces(image_path: str, save_annotated: bool = True) -> list[dict]:
    """Detect faces in an image using Haar cascade classifier.

    Args:
        image_path: Path to the input image.
        save_annotated: If True, save a copy with bounding boxes drawn (default True).

    Returns:
        List of dicts with x, y, w, h for each detected face.
    """
    img, err = validate_image(image_path, "detect_faces")
    if err:
        return err

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_list = []
    for (x, y, w, h) in faces:
        face_list.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        if save_annotated:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result = {"face_count": len(face_list), "faces": face_list}

    if save_annotated and len(face_list) > 0:
        out = output_path(image_path, "faces")
        cv2.imwrite(out, img)
        result["annotated_image"] = out

    return success_response("detect_faces", result, suggestions=[
        "Run analyze_image to get color and brightness stats for this image",
        "Run detect_edges to see the structural outlines in this image",
    ])


if __name__ == "__main__":
    mcp.run()
