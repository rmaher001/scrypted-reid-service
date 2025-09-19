import scrypted_sdk
from scrypted_sdk import ScryptedDeviceBase, BufferConverter
import json
import os
import urllib.request

class ReIDService(ScryptedDeviceBase, BufferConverter):
    def __init__(self, nativeId=None):
        super().__init__(nativeId)
        print("ReIDService initialized - using BufferConverter interface", flush=True)
        # Set the converter properties for BufferConverter interface
        self.fromMimeType = "application/json"
        self.toMimeType = "application/reid"

        # Initialize ReID engine lazily
        self.reid_engine = None
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
                import time
                init_start = time.time()
                print("🚀 ReID service initializing...", flush=True)

                print("🔧 Testing basic imports...", flush=True)
                import numpy as np
                print(f"✅ numpy imported successfully: {np.__version__}", flush=True)

                import onnxruntime as ort
                print(f"✅ onnxruntime imported successfully: {ort.__version__}", flush=True)

                # Test model download
                print("🔧 Testing model download...", flush=True)
                download_start = time.time()
                model_path = self.download_model()
                download_time = time.time() - download_start
                print(f"⏱️  Model download took {download_time:.1f}s", flush=True)

                if model_path and os.path.exists(model_path):
                    print(f"✅ Model available at: {model_path}", flush=True)

                    # Test ONNX model loading
                    print("🔧 Loading ONNX model into inference session...", flush=True)
                    session_start = time.time()
                    session = ort.InferenceSession(model_path)
                    session_time = time.time() - session_start
                    print(f"⏱️  ONNX session creation took {session_time:.1f}s", flush=True)

                    # Print model info
                    input_info = session.get_inputs()[0]
                    output_info = session.get_outputs()[0]
                    print(f"✅ Model loaded - Input: {input_info.name} {input_info.shape}, Output: {output_info.name} {output_info.shape}", flush=True)

                    self.reid_engine = "ready_for_reid_integration"
                    total_time = time.time() - init_start
                    print(f"✅ ReID service fully initialized in {total_time:.1f}s - ready for processing!", flush=True)
                else:
                    print("❌ Model download failed", flush=True)
                    self.reid_engine = "model_download_failed"

            except ImportError as e:
                print(f"❌ Import error: {e}", flush=True)
                self.reid_engine = "dependencies_failed"
            except Exception as e:
                print(f"❌ Unexpected error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self.reid_engine = "initialization_failed"

        if fromMimeType == "application/json" and toMimeType == "application/reid":
            # Parse the JSON data
            detections_data = json.loads(data) if isinstance(data, str) else data

            # Log received data
            device_name = detections_data.get('deviceName', 'unknown')
            detection_count = detections_data.get('detectionCount', 0)
            has_persons = detections_data.get('hasPersons', False)
            timestamp = detections_data.get('timestamp')

            print(f"ReID: {device_name} - {detection_count} detections, hasPersons: {has_persons}", flush=True)

            # If no detections, return early
            if detection_count == 0:
                return json.dumps({
                    "processed": False,
                    "timestamp": timestamp,
                    "deviceName": device_name,
                    "message": "No detections to process",
                    "isNew": False,
                    "personId": None
                })

            # Process person detections
            person_detections = []
            if detections_data.get('detections'):
                for det in detections_data['detections']:
                    if det.get('className') in ['person', 'face']:
                        person_detections.append({
                            'className': det.get('className'),
                            'label': det.get('label'),
                            'score': det.get('score'),
                            'boundingBox': det.get('boundingBox'),
                            'id': det.get('id')
                        })
                        print(f"  - {det.get('className')}: {det.get('label') or 'unknown'} (score: {det.get('score', 0):.2f})", flush=True)

            # Basic processing (will add ReID logic once dependencies confirmed working)
            if self.reid_engine == "dependencies_working":
                status = "Dependencies OK, ready for ReID integration"
            else:
                status = "Dependencies failed, using fallback"

            result = {
                "processed": True,
                "timestamp": timestamp,
                "deviceName": device_name,
                "personCount": len(person_detections),
                "persons": person_detections,
                "isNew": True,  # Will be determined by actual ReID once integrated
                "personId": None,  # Will be assigned by actual ReID once integrated
                "message": f"{status}: {len(person_detections)} person detections"
            }

            return json.dumps(result)

        return data

def create_scrypted_plugin():
    return ReIDService()