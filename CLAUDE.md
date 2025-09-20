# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **backend ReID processing service** for the Smart Notifier system located at `~/node/scrypted/plugins/smart-notifier`. It's part of a two-plugin architecture that provides cross-camera person deduplication for smart notifications.

**Current Status (2025-09-19)**: ⚠️ **BLOCKED** - Python plugin deployment crashes Scrypted server
- Smart Notifier (TypeScript frontend) is working
- ReID Service (Python backend) crashes during deployment
- Issue: Large ONNX model (8.3MB) or Python runtime stability problems

This Python-based service uses ONNX Runtime and the OSNet AIN model to identify and track persons across multiple cameras within a 60-second window, preventing duplicate notifications for the same person.

## Architecture

### Two-Plugin System

This ReID Service works with **@scrypted/smart-notifier** (TypeScript frontend):

1. **Smart Notifier** - Camera mixin that captures ObjectDetector events
2. **ReID Service** - This backend service that processes person re-identification
3. **Communication** - RPC via BufferConverter interface (`application/json` → `application/reid`)

### Core Components

- **main.py**: Primary plugin entry point implementing ScryptedDeviceBase and BufferConverter interfaces
- **reid_engine.py**: Core ReID processing engine with ONNX model inference and person tracking
- **Python Runtime**: Plugin runs in Python (specified in package.json "runtime": "python")
- **ONNX Model**: Uses OSNet AIN model for 512-dimensional person embeddings downloaded at runtime

### Data Flow (Smart Notifier → ReID Service)

1. Smart Notifier captures ObjectDetector events from cameras
2. Filters for person/face detections and retrieves snapshots via `getRecordingStreamThumbnail()`
3. Sends detection data with base64-encoded images to ReID Service
4. ReID Service crops person bounding boxes from full images
5. Cropped images processed through ONNX model to extract 512-dim embeddings
6. Embeddings compared against tracked persons using cosine similarity (threshold: 0.6)
7. Returns deduplication result (`isNew`, `personId`, `matchedCameras`) to Smart Notifier
8. Smart Notifier decides whether to send notification based on `isNew` flag

## Development Commands

### Build and Deploy
```bash
npm run build                    # Build TypeScript (though main code is Python)
npm run scrypted-deploy-debug   # Deploy to debug environment
./deploy.sh [server_ip]         # Deploy to specific server (defaults to 192.168.86.74)
```

### Development Workflow
```bash
npm install                     # Install dependencies
npm run build                   # Build project
npm run scrypted-deploy-debug   # Deploy for testing
```

## Python Dependencies

Dependencies are installed automatically on first run via requirements.txt:
- numpy
- onnxruntime
- opencv-python-headless
- Pillow

## Model Management

The ONNX model (osnet_ain_multisource.onnx) is downloaded automatically from GitHub releases on first initialization. It's stored in the Scrypted plugin volume directory for persistence.

## Configuration

### ReID Engine Settings
- **Similarity Threshold**: 0.6 (minimum cosine similarity to consider same person)
- **Tracking Window**: 60 seconds (how long to remember persons)
- **Feature Dimensions**: 512 (OSNet embedding size)
- **Input Resolution**: 256x128 (required by OSNet model)

### Deployment Settings
- Default deployment target: 192.168.86.74:10443
- Plugin integrates as mixin to cameras with object detection
- Configuration through Scrypted UI after deployment

## File Structure Notes

- **src/**: Python source code
- **dist/**: Built TypeScript artifacts (legacy from original TS implementation)
- **src_typescript_backup/**: Original TypeScript implementation backup
- **models/**: Directory for ONNX model storage
- **out/**: Plugin build output

## Development Notes

- Plugin uses lazy initialization - ReID engine loads only when first detection arrives
- Extensive logging for debugging (flush=True for real-time output in Scrypted)
- Error handling preserves functionality even if ReID fails
- Person tracking uses LRU cache with time-based expiration
- Support for both 'person' and 'face' detection types from upstream

## Current Issues & Debugging

### Known Problems
- **Plugin crashes Scrypted server on deployment** (primary blocker)
- Large ONNX model (8.3MB) may be causing bundle size issues
- Python runtime stability problems in Scrypted environment

### Related Files
- Main Smart Notifier frontend: `~/node/scrypted/plugins/smart-notifier/`
- Architecture documentation: `~/node/scrypted/plugins/smart-notifier/ARCHITECTURE.md`

### Deployment Testing
The `deploy.sh` script targets a specific test server (192.168.86.74:10443) for debugging deployment issues.

### Remote Server Debugging
- Plugin deploys to remote Scrypted server at 192.168.86.74:10443
- All plugin issues (crashes, errors, runtime problems) occur on the remote server
- **Debugging workflow**: Copy any relevant logs from remote server and paste into local `reid-service.log` file for analysis
- Local `reid-service.log` serves as the "inbox" for remote server diagnostics
- Current crash pattern: "Error: close" immediately during Python plugin initialization