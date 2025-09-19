/**
 * ReID Engine - Core person re-identification using ONNX
 *
 * Uses OSNet AIN model to generate 512-dimensional embeddings
 * for person matching across cameras.
 */

import * as ort from 'onnxruntime-node';
import { LRUCache } from 'lru-cache';
import * as path from 'path';
import sharp from 'sharp';

const FEATURE_DIM = 512;
const INPUT_HEIGHT = 256;
const INPUT_WIDTH = 128;

// ImageNet normalization constants
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

interface TrackedPerson {
    personId: string;
    firstSeen: number;
    lastSeen: number;
    embedding: Float32Array;
    cameraId: string;
    cameraName: string;
    snapshot?: Buffer;  // Original snapshot from first detection
}

export class ReIDEngine {
    private session: ort.InferenceSession | null = null;
    private trackedPersons: LRUCache<string, TrackedPerson>;
    private trackingWindowMs = 60000; // 60 seconds default
    private debugMode = false;
    private similarityThreshold = 0.6; // Minimum similarity to consider same person

    constructor() {
        // LRU cache automatically expires old entries
        this.trackedPersons = new LRUCache<string, TrackedPerson>({
            max: 1000, // Maximum tracked persons
            ttl: this.trackingWindowMs,
            updateAgeOnGet: false,
            updateAgeOnHas: false
        });
    }

    async initialize(): Promise<void> {
        const modelPath = path.join(__dirname, '..', 'models', 'osnet_ain_multisource.onnx');

        try {
            this.session = await ort.InferenceSession.create(modelPath);
            this.log('ReID engine initialized with OSNet ONNX model');
        } catch (error) {
            throw new Error(`Failed to load ONNX model: ${error}`);
        }
    }

    /**
     * Process a detection and determine if it's a new person
     * @returns { isNew: boolean, personId: string, firstDetection?: TrackedPerson }
     */
    async processDetection(
        imageBuffer: Buffer,
        cameraId: string,
        cameraName: string,
        detectionType: string
    ): Promise<{
        isNew: boolean;
        personId: string;
        firstDetection?: TrackedPerson;
    }> {
        if (!this.session) {
            throw new Error('ReID engine not initialized');
        }

        // Only process person/face detections for ReID
        if (detectionType !== 'person' && detectionType !== 'face') {
            // For vehicles/animals, use simple time-based deduplication per camera
            const simpleId = `${detectionType}_${cameraId}_${Math.floor(Date.now() / this.trackingWindowMs)}`;
            return {
                isNew: !this.trackedPersons.has(simpleId),
                personId: simpleId
            };
        }

        // Extract embedding from image
        const embedding = await this.extractEmbedding(imageBuffer);

        // Find best match in tracked persons
        const match = this.findBestMatch(embedding);

        const now = Date.now();

        if (match && match.similarity >= this.similarityThreshold) {
            // Known person - update last seen
            const tracked = this.trackedPersons.get(match.personId)!;
            tracked.lastSeen = now;

            this.log(`ReID match: ${match.personId} on ${cameraName} (similarity: ${match.similarity.toFixed(3)})`);

            return {
                isNew: false,
                personId: match.personId,
                firstDetection: tracked
            };
        } else {
            // New person - create tracking entry
            const personId = this.generatePersonId();

            const tracked: TrackedPerson = {
                personId,
                firstSeen: now,
                lastSeen: now,
                embedding,
                cameraId,
                cameraName,
                snapshot: imageBuffer
            };

            this.trackedPersons.set(personId, tracked);

            this.log(`New person: ${personId} detected on ${cameraName}`);

            return {
                isNew: true,
                personId,
                firstDetection: tracked
            };
        }
    }

    /**
     * Extract ReID embedding from image
     */
    private async extractEmbedding(imageBuffer: Buffer): Promise<Float32Array> {
        if (!this.session) {
            throw new Error('Session not initialized');
        }

        // Preprocess image: resize to 256x128 and normalize
        const tensor = await this.preprocessImage(imageBuffer);

        // Run inference
        const feeds = { input: tensor };
        const results = await this.session.run(feeds);

        // Get embedding (already L2-normalized by model)
        const output = results.output;
        const embedding = new Float32Array(output.data as Float32Array);

        // Verify it's normalized
        const norm = this.calculateNorm(embedding);
        if (Math.abs(norm - 1.0) > 0.01) {
            this.log(`Warning: Embedding norm is ${norm}, expected ~1.0`);
        }

        return embedding;
    }

    /**
     * Preprocess image for ReID model
     */
    private async preprocessImage(imageBuffer: Buffer): Promise<ort.Tensor> {
        // Resize to 256x128 using sharp
        const resized = await sharp(imageBuffer)
            .resize(INPUT_WIDTH, INPUT_HEIGHT, {
                fit: 'fill',
                kernel: 'lanczos3'
            })
            .removeAlpha()
            .raw()
            .toBuffer();

        // Convert to RGB tensor with ImageNet normalization
        const pixelCount = INPUT_HEIGHT * INPUT_WIDTH;
        const tensorData = new Float32Array(3 * pixelCount);

        // Sharp outputs in RGB order
        for (let i = 0; i < pixelCount; i++) {
            const pixelIndex = i * 3;

            // R channel
            tensorData[0 * pixelCount + i] =
                (resized[pixelIndex] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];

            // G channel
            tensorData[1 * pixelCount + i] =
                (resized[pixelIndex + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];

            // B channel
            tensorData[2 * pixelCount + i] =
                (resized[pixelIndex + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }

        // Create ONNX tensor (batch_size=1, channels=3, height=256, width=128)
        return new ort.Tensor('float32', tensorData, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
    }

    /**
     * Find best matching person in cache
     */
    private findBestMatch(embedding: Float32Array): { personId: string; similarity: number } | null {
        let bestMatch: { personId: string; similarity: number } | null = null;
        let bestSimilarity = 0;

        for (const [personId, tracked] of this.trackedPersons.entries()) {
            const similarity = this.cosineSimilarity(embedding, tracked.embedding);

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = { personId, similarity };
            }
        }

        return bestMatch;
    }

    /**
     * Calculate cosine similarity between embeddings
     * Note: Since embeddings are L2-normalized, this is just dot product
     */
    private cosineSimilarity(a: Float32Array, b: Float32Array): number {
        let dotProduct = 0;
        for (let i = 0; i < FEATURE_DIM; i++) {
            dotProduct += a[i] * b[i];
        }
        return dotProduct;
    }

    /**
     * Calculate L2 norm of vector
     */
    private calculateNorm(vec: Float32Array): number {
        let sum = 0;
        for (let i = 0; i < vec.length; i++) {
            sum += vec[i] * vec[i];
        }
        return Math.sqrt(sum);
    }

    /**
     * Generate unique person ID
     */
    private generatePersonId(): string {
        return `person_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Set tracking window duration
     */
    setTrackingWindow(ms: number): void {
        this.trackingWindowMs = ms;
        // Update cache TTL
        this.trackedPersons = new LRUCache<string, TrackedPerson>({
            max: 1000,
            ttl: ms,
            updateAgeOnGet: false,
            updateAgeOnHas: false
        });
    }

    /**
     * Enable/disable debug logging
     */
    setDebugMode(enabled: boolean): void {
        this.debugMode = enabled;
    }

    /**
     * Get current stats
     */
    getStats(): {
        trackedPersons: number;
        cacheSize: number;
    } {
        return {
            trackedPersons: this.trackedPersons.size,
            cacheSize: this.trackedPersons.size * FEATURE_DIM * 4 // Approximate bytes
        };
    }

    /**
     * Clear all tracked persons
     */
    clear(): void {
        this.trackedPersons.clear();
        this.log('Cleared all tracked persons');
    }

    private log(message: string): void {
        if (this.debugMode) {
            console.log(`[ReID] ${message}`);
        }
    }
}