"""
ReID Engine - Core person re-identification using ONNX

Uses OSNet AIN model to generate 512-dimensional embeddings
for person matching across cameras.
"""

import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

FEATURE_DIM = 512
INPUT_HEIGHT = 256
INPUT_WIDTH = 128

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class TrackedPerson:
    def __init__(self, person_id: str, embedding: np.ndarray, camera_id: str,
                 camera_name: str, snapshot: bytes = None):
        self.person_id = person_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.created_at = time.time()
        self.embedding = embedding
        self.first_camera_id = camera_id
        self.first_camera_name = camera_name
        self.cameras = [camera_id]  # List of camera IDs that have seen this person
        self.camera_names = [camera_name]  # Corresponding camera names
        self.snapshot = snapshot

class ReIDEngine:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.tracked_persons: OrderedDict[str, TrackedPerson] = OrderedDict()
        self.tracking_window_ms = 60000  # 60 seconds default
        self.debug_mode = False
        self.similarity_threshold = 0.7  # Minimum similarity to consider same person

        # Basic stats
        self.total_requests = 0
        self.start_time = time.time()

    async def initialize(self):
        """Initialize ONNX model"""
        model_path = os.path.join(os.path.dirname(__file__), 'osnet_ain_multisource.onnx')

        try:
            self.session = ort.InferenceSession(model_path)
            self.log('ReID engine initialized with OSNet ONNX model')
        except Exception as e:
            raise Exception(f'Failed to load ONNX model: {e}')

    async def process_detection(self, image_buffer: bytes, camera_id: str,
                               camera_name: str, detection_type: str) -> Dict:
        """
        Process a detection and determine if it's a new person
        Returns: { isNew: bool, personId: str, firstDetection?: TrackedPerson }
        """
        if not self.session:
            raise Exception('ReID engine not initialized')

        # Update stats
        self.total_requests += 1

        # Clean up old entries
        self._cleanup_old_entries()

        # Only process person/face detections for ReID
        if detection_type not in ['person', 'face']:
            # For vehicles/animals, use simple time-based deduplication per camera
            simple_id = f'{detection_type}_{camera_id}_{int(time.time() * 1000 / self.tracking_window_ms)}'
            is_new = simple_id not in self.tracked_persons
            if is_new:
                # Create a dummy entry to track it
                self.tracked_persons[simple_id] = TrackedPerson(
                    simple_id, np.zeros(FEATURE_DIM), camera_id, camera_name
                )
            return {
                'isNew': is_new,
                'personId': simple_id
            }

        # Extract embedding from image
        embedding = await self._extract_embedding(image_buffer)

        # Find best match in tracked persons
        match = self._find_best_match(embedding)

        if match and match[1] >= self.similarity_threshold:
            # Known person - update last seen and camera tracking
            person_id = match[0]
            tracked = self.tracked_persons[person_id]
            tracked.last_seen = time.time()

            # Add camera to tracking if not already seen
            if camera_id not in tracked.cameras:
                tracked.cameras.append(camera_id)
                tracked.camera_names.append(camera_name)

            self.log(f'Match on {camera_name} (similarity: {match[1]:.3f})', person_id, False)

            return {
                'isNew': False,
                'personId': person_id,
                'confidence': float(match[1]),
                'matchedCameras': tracked.camera_names.copy(),
                'embedding': embedding.tolist(),
                'firstDetection': self._tracked_to_dict(tracked)
            }
        else:
            # New person - create tracking entry
            person_id = self._generate_person_id()

            tracked = TrackedPerson(
                person_id,
                embedding,
                camera_id,
                camera_name,
                image_buffer
            )

            self.tracked_persons[person_id] = tracked

            self.log(f'New person detected on {camera_name}', person_id, True)

            return {
                'isNew': True,
                'personId': person_id,
                'confidence': 1.0,  # New person has 100% confidence
                'matchedCameras': [camera_name],
                'embedding': embedding.tolist(),
                'firstDetection': self._tracked_to_dict(tracked)
            }

    async def _extract_embedding(self, image_buffer: bytes) -> np.ndarray:
        """Extract ReID embedding from image"""
        if not self.session:
            raise Exception('Session not initialized')

        # Preprocess image
        tensor = self._preprocess_image(image_buffer)

        # Run inference
        outputs = self.session.run(['output'], {'input': tensor})

        # Get embedding (already L2-normalized by model)
        embedding = outputs[0].squeeze()

        # Verify it's normalized
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 0.01:
            self.log(f'Warning: Embedding norm is {norm}, expected ~1.0')

        return embedding

    def _preprocess_image(self, image_buffer: bytes) -> np.ndarray:
        """Preprocess image for ReID model"""
        # Load and resize image
        img = Image.open(io.BytesIO(image_buffer))
        img = img.convert('RGB')
        img = img.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # Transpose to CHW format
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def _find_best_match(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find best matching person in cache"""
        best_match = None
        best_similarity = 0

        for person_id, tracked in self.tracked_persons.items():
            # Skip non-person entries
            if not isinstance(tracked.embedding, np.ndarray) or tracked.embedding.shape[0] != FEATURE_DIM:
                continue

            similarity = self._cosine_similarity(embedding, tracked.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (person_id, similarity)

        return best_match

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Since embeddings are L2-normalized, this is just dot product
        return float(np.dot(a, b))

    def _generate_person_id(self) -> str:
        """Generate unique person ID"""
        import random
        import string
        rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
        return f'person_{int(time.time() * 1000)}_{rand_str}'

    def _cleanup_old_entries(self):
        """Remove entries older than cache expiry (tracking window + 30s buffer)"""
        current_time = time.time()
        # Cache expiry is 90 seconds (60s tracking window + 30s buffer)
        cache_expiry_ms = 90000
        cutoff_time = current_time - (cache_expiry_ms / 1000)

        # Find keys to remove
        keys_to_remove = []
        for person_id, tracked in self.tracked_persons.items():
            if tracked.last_seen < cutoff_time:
                keys_to_remove.append(person_id)

        # Remove old entries
        for key in keys_to_remove:
            del self.tracked_persons[key]

    def _tracked_to_dict(self, tracked: TrackedPerson) -> Dict:
        """Convert TrackedPerson to dictionary"""
        return {
            'personId': tracked.person_id,
            'firstSeen': tracked.first_seen * 1000,  # Convert to ms
            'lastSeen': tracked.last_seen * 1000,
            'createdAt': tracked.created_at * 1000,
            'firstCameraId': tracked.first_camera_id,
            'firstCameraName': tracked.first_camera_name,
            'cameras': tracked.cameras.copy(),
            'cameraNames': tracked.camera_names.copy(),
            'snapshot': tracked.snapshot
        }

    def set_tracking_window(self, ms: int):
        """Set tracking window duration"""
        self.tracking_window_ms = ms
        # Clean up based on new window
        self._cleanup_old_entries()

    def set_debug_mode(self, enabled: bool):
        """Enable/disable debug logging"""
        self.debug_mode = enabled

    def get_stats(self) -> Dict:
        """Get current stats"""
        uptime_seconds = time.time() - self.start_time
        requests_per_second = self.total_requests / uptime_seconds if uptime_seconds > 0 else 0

        return {
            'trackedPersons': len(self.tracked_persons),
            'cacheSize': len(self.tracked_persons) * FEATURE_DIM * 4,  # Approximate bytes
            'totalRequests': self.total_requests,
            'requestsPerSecond': round(requests_per_second, 2),
            'uptimeSeconds': round(uptime_seconds, 1)
        }

    def clear(self):
        """Clear all tracked persons"""
        self.tracked_persons.clear()
        self.log('Cleared all tracked persons')

    def log(self, message: str, person_id: str = None, is_new: bool = None):
        """Log message if debug mode is enabled"""
        if self.debug_mode:
            timestamp = int(time.time() * 1000)
            if person_id:
                is_new_str = "NEW " if is_new else "SEEN" if is_new is False else "    "
                print(f'[{timestamp}] [{person_id:>20}] [{is_new_str:>4}] [ReID] {message}', flush=True)
            else:
                print(f'[{timestamp}] {" " * 22} {" " * 6} [ReID] {message}', flush=True)