# OpenCV Vision MCP Server - Incremental Plan

## Phase 1: Project Setup
- [ ] Set up Python project structure (`opencv_mcp_server.py`)
- [ ] Install dependencies: `mcp`, `opencv-python`, `numpy`
- [ ] Create a basic MCP server skeleton using FastMCP
- [ ] Add a test image to the project for development/testing

## Phase 2: Tool 1 - Edge Detection
- [ ] Implement `detect_edges(image_path: str) -> str`
  - Read image with `cv2.imread`
  - Apply Canny edge detection (`cv2.Canny`)
  - Save output image to a generated path (avoid overwriting)
  - Return the output path
- [ ] Test manually by running the server and calling the tool

## Phase 3: Tool 2 - Image Stats
- [ ] Implement `analyze_image(image_path: str) -> dict`
  - Compute resolution (width x height)
  - Compute mean color per channel (BGR)
  - Compute overall brightness (mean of grayscale conversion)
  - Return results as a dictionary
- [ ] Test manually

## Phase 4: Tool 3 - Face Detection
- [ ] Implement `detect_faces(image_path: str) -> list[dict]`
  - Load Haar cascade classifier (`haarcascade_frontalface_default.xml`)
  - Convert image to grayscale
  - Run `detectMultiScale` to find faces
  - Optionally save annotated image with bounding boxes drawn
  - Return list of `{"x", "y", "w", "h"}` dicts
- [ ] Test with an image containing faces

## Phase 5: Integration & Polish
- [ ] Add input validation (check file exists, supported formats)
- [ ] Use unique output filenames (e.g., timestamp or hash-based) to avoid collisions
- [ ] Add proper error messages for missing/corrupt images
- [ ] Test all three tools end-to-end with the MCP server running

## Phase 6 (Optional): Extensions
- [ ] **Image similarity** - compare two images via histogram correlation
- [ ] **TennisVision Lite** - `analyze_swing(image_path)` using MediaPipe pose estimation for joint angles and posture feedback
- [ ] **Object detection** - integrate a pretrained YOLO model for general object detection
