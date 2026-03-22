import os
import cv2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("opencv-vision")


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
    img = cv2.imread(image_path)
    if img is None:
        return f"Error: could not read image at {image_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join("output", f"{base}_edges.jpg")
    cv2.imwrite(output_path, edges)
    return output_path


@mcp.tool()
def analyze_image(image_path: str) -> dict:
    """Analyze an image and return statistics.

    Args:
        image_path: Path to the input image.

    Returns:
        Dictionary with width, height, mean color per channel (BGR), and brightness.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"could not read image at {image_path}"}

    height, width = img.shape[:2]
    mean_bgr = img.mean(axis=(0, 1)).tolist()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    return {
        "width": width,
        "height": height,
        "mean_blue": round(mean_bgr[0], 2),
        "mean_green": round(mean_bgr[1], 2),
        "mean_red": round(mean_bgr[2], 2),
        "brightness": round(brightness, 2),
    }


@mcp.tool()
def detect_faces(image_path: str, save_annotated: bool = True) -> list[dict]:
    """Detect faces in an image using Haar cascade classifier.

    Args:
        image_path: Path to the input image.
        save_annotated: If True, save a copy with bounding boxes drawn (default True).

    Returns:
        List of dicts with x, y, w, h for each detected face.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [{"error": f"could not read image at {image_path}"}]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        if save_annotated:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save_annotated and len(results) > 0:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join("output", f"{base}_faces.jpg")
        cv2.imwrite(output_path, img)
        results.append({"annotated_image": output_path})

    return results


if __name__ == "__main__":
    mcp.run()
