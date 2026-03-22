# vision-mcp
MCP server exposing OpenCV computer vision tools for LLM workflows

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate test image

```bash
python3 create_test_image.py
```

This creates `test_image.jpg` with colored shapes for testing.

## Run the MCP server

```bash
python3 opencv_mcp_server.py
```
