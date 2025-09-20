import scrypted_sdk
from scrypted_sdk import ScryptedDeviceBase, BufferConverter
import json
import os
import urllib.request
import base64
import io
import time
from typing import Optional

def log_with_timestamp(message, person_id=None, is_new=None):
    """Log message with millisecond timestamp, person ID, and isNew columns"""
    timestamp = int(time.time() * 1000)
    if person_id:
        is_new_str = "NEW " if is_new else "SEEN" if is_new is False else "    "
        print(f"[{timestamp}] [{person_id:>20}] [{is_new_str:>4}] {message}", flush=True)
    else:
        print(f"[{timestamp}] {' ' * 22} {' ' * 6} {message}", flush=True)

class ReIDService(ScryptedDeviceBase, BufferConverter):
    def __init__(self, nativeId=None):
        super().__init__(nativeId)
        print("ReIDService initialized - using BufferConverter interface", flush=True)
        # Set the converter properties for BufferConverter interface
        self.fromMimeType = "application/json"
        self.toMimeType = "application/reid"

        # Initialize ReID engine lazily
        self.reid_engine = None
        self.onnx_session = None
        self.last_stats_log = 0
        print("ReIDService constructor completed successfully", flush=True)

    def download_model(self):
        """Download ONNX model using Scrypted's standard pattern"""
        try:
            # Use Scrypted's plugin volume directory
            files_path = os.path.join(os.environ.get("SCRYPTED_PLUGIN_VOLUME", "."), "files")
            model_path = os.path.join(files_path, "osnet_ain_multisource.onnx")

            if os.path.exists(model_path):
                print(f"✅ Model already exists at {model_path}", flush=True)
                return model_path

            print("📥 ONNX model not found, downloading...", flush=True)
            model_url = "https://github.com/rmaher001/scrypted-smart-notifier/releases/download/v1.0/osnet_ain_multisource.onnx"

            # Create files directory
            os.makedirs(files_path, exist_ok=True)
            tmp = model_path + ".tmp"

            print(f"Downloading from {model_url}...", flush=True)
            response = urllib.request.urlopen(model_url)
            if response.getcode() < 200 or response.getcode() >= 300:
                raise Exception(f"HTTP {response.getcode()}")

            read = 0
            with open(tmp, "wb") as f:
                while True:
                    data = response.read(1024 * 1024)  # 1MB chunks
                    if not data:
                        break
                    read += len(data)
                    f.write(data)

            os.rename(tmp, model_path)
            print(f"✅ Model downloaded successfully: {model_path} ({read} bytes)", flush=True)
            return model_path

        except Exception as e:
            print(f"❌ Failed to download model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    async def convert(self, data, fromMimeType, toMimeType, options=None):
        """BufferConverter interface for ReID processing"""
        print(f"ReID convert called: from={fromMimeType}, to={toMimeType}", flush=True)

        # Lazy initialization of ReID engine
        if self.reid_engine is None:
            try:
                init_start = time.time()
                print("🚀 ReID service initializing...", flush=True)

                # Import required modules
                print("🔧 Importing modules...", flush=True)
                import numpy as np
                print(f"✅ numpy imported: {np.__version__}", flush=True)

                import onnxruntime as ort
                print(f"✅ onnxruntime imported: {ort.__version__}", flush=True)

                # Download model if needed
                print("🔧 Checking model...", flush=True)
                download_start = time.time()
                model_path = self.download_model()
                download_time = time.time() - download_start
                print(f"⏱️  Model check took {download_time:.1f}s", flush=True)

                if model_path and os.path.exists(model_path):
                    print(f"✅ Model available at: {model_path}", flush=True)

                    # Initialize ONNX session for direct use
                    print("🔧 Loading ONNX model...", flush=True)
                    session_start = time.time()
                    self.onnx_session = ort.InferenceSession(model_path)
                    session_time = time.time() - session_start
                    print(f"⏱️  ONNX session creation took {session_time:.1f}s", flush=True)

                    # Print model info
                    input_info = self.onnx_session.get_inputs()[0]
                    output_info = self.onnx_session.get_outputs()[0]
                    print(f"✅ Model loaded - Input: {input_info.name} {input_info.shape}, Output: {output_info.name} {output_info.shape}", flush=True)

                    # Now import ReIDEngine after dependencies are confirmed working
                    print("🔧 Importing ReIDEngine...", flush=True)
                    from reid_engine import ReIDEngine
                    print("✅ ReIDEngine imported", flush=True)

                    # Create ReIDEngine instance
                    self.reid_engine = ReIDEngine()
                    self.reid_engine.session = self.onnx_session
                    self.reid_engine.debug_mode = True

                    total_time = time.time() - init_start
                    print(f"✅ ReID service fully initialized in {total_time:.1f}s!", flush=True)
                else:
                    print("❌ Model download failed", flush=True)
                    self.reid_engine = None
                    return data

            except ImportError as e:
                print(f"❌ Import error: {e}", flush=True)
                self.reid_engine = None
                return data
            except Exception as e:
                print(f"❌ Unexpected error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.reid_engine = None
                return data

        if fromMimeType == "application/json" and toMimeType == "application/reid":
            # Parse the JSON data
            detections_data = json.loads(data) if isinstance(data, str) else data

            # Log received data
            device_name = detections_data.get('deviceName', 'unknown')
            device_id = detections_data.get('deviceId', 'unknown')
            detection_count = detections_data.get('detectionCount', 0)
            has_persons = detections_data.get('hasPersons', False)
            timestamp = detections_data.get('timestamp')
            image_base64 = detections_data.get('imageBase64')

            print(f"ReID: {device_name} - {detection_count} detections, hasPersons: {has_persons}", flush=True)

            # Log stats periodically (every 60 seconds)
            current_time = time.time()
            if self.reid_engine and current_time - self.last_stats_log > 60:
                stats = self.reid_engine.get_stats()
                print(f"ReID Stats: {stats['totalRequests']} requests, {stats['requestsPerSecond']} req/s, {stats['trackedPersons']} persons tracked", flush=True)
                self.last_stats_log = current_time

            # If no detections or no ReID engine, return early
            if detection_count == 0 or not self.reid_engine:
                return json.dumps({
                    "processed": False,
                    "timestamp": timestamp,
                    "deviceName": device_name,
                    "message": "No detections to process or ReID not ready",
                    "isNew": False,
                    "personId": None
                })

            # Process person detections
            person_detections = []
            reid_results = []

            if detections_data.get('detections') and image_base64:
                # Decode the image once
                try:
                    image_bytes = base64.b64decode(image_base64)
                    print(f"Decoded image size: {len(image_bytes)} bytes", flush=True)

                    # Import PIL here when we need it
                    from PIL import Image

                    # Open image with PIL for cropping
                    full_image = Image.open(io.BytesIO(image_bytes))
                    img_width, img_height = full_image.size
                    print(f"Image dimensions: {img_width}x{img_height}", flush=True)
                except Exception as e:
                    print(f"Error decoding image: {e}", flush=True)
                    full_image = None

                for det in detections_data['detections']:
                    if det.get('className') in ['person', 'face']:
                        face_label = det.get('faceLabel', det.get('label'))

                        person_det = {
                            'className': det.get('className'),
                            'label': face_label,
                            'score': det.get('score'),
                            'boundingBox': det.get('boundingBox'),
                            'id': det.get('id')
                        }
                        person_detections.append(person_det)

                        print(f"  - {det.get('className')}: {face_label or 'unknown'} (score: {det.get('score', 0):.2f})", flush=True)

                        # Process with ReID if we have image and bounding box
                        if full_image and det.get('boundingBox') and det.get('className') == 'person':
                            try:
                                # Crop person from image
                                bbox = det['boundingBox']
                                # bbox format: [x, y, width, height]
                                x, y, w, h = bbox

                                # Ensure bounds are within image
                                x = max(0, min(x, img_width))
                                y = max(0, min(y, img_height))
                                x2 = min(x + w, img_width)
                                y2 = min(y + h, img_height)

                                cropped = full_image.crop((x, y, x2, y2))
                                print(f"Cropped person: {x},{y} to {x2},{y2} (size: {cropped.size})", flush=True)

                                # Convert cropped image to bytes
                                crop_buffer = io.BytesIO()
                                cropped.save(crop_buffer, format='JPEG')
                                crop_buffer.seek(0)
                                crop_bytes = crop_buffer.getvalue()

                                # Process with ReID engine
                                reid_result = await self.reid_engine.process_detection(
                                    crop_bytes,
                                    device_id,
                                    device_name,
                                    'person'
                                )

                                log_with_timestamp(f"confidence={reid_result.get('confidence', 0):.3f}", reid_result.get('personId'), reid_result.get('isNew'))

                                # Add ReID result to detection
                                person_det['personId'] = reid_result.get('personId')
                                person_det['isNew'] = reid_result.get('isNew')
                                person_det['confidence'] = reid_result.get('confidence')
                                person_det['matchedCameras'] = reid_result.get('matchedCameras')
                                person_det['embedding'] = reid_result.get('embedding')
                                reid_results.append(reid_result)

                            except Exception as e:
                                print(f"Error processing ReID for detection: {e}", flush=True)
                                import traceback
                                traceback.print_exc()

            # Determine overall result
            is_new_person = any(r.get('isNew', False) for r in reid_results) if reid_results else False
            person_ids = [r.get('personId') for r in reid_results if r.get('personId')]

            result = {
                "processed": True,
                "timestamp": timestamp,
                "deviceName": device_name,
                "deviceId": device_id,
                "personCount": len(person_detections),
                "persons": person_detections,
                "isNew": is_new_person,
                "personIds": person_ids,
                "message": f"Processed {len(person_detections)} person detections"
            }

            return json.dumps(result)

        return data

def create_scrypted_plugin():
    print("Creating ReID Service plugin instance", flush=True)
    return ReIDService()

async def fork():
    print("Forking ReID Service for isolated ONNX loading", flush=True)
    return ReIDService()