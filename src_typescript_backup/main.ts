import {
    Device, DeviceProvider, MixinProvider, ObjectDetection, ObjectDetectionTypes,
    ObjectsDetected, ScryptedDevice, ScryptedDeviceBase, ScryptedDeviceType,
    ScryptedInterface, Setting, Settings, SettingValue, MediaObject,
    EventListenerRegister, EventDetails, Notifier, NotifierOptions,
    WritableDeviceState, ObjectDetectionModel, VideoFrame,
    ObjectDetectionGeneratorSession, Logger, VideoRecorder
} from '@scrypted/sdk';
import { ReIDEngine } from './reid-engine';

import sdk from '@scrypted/sdk';
const { deviceManager, systemManager, mediaManager } = sdk;
const log = sdk.log as Logger | undefined;

const TRACKING_WINDOW_DEFAULT = 60; // seconds

class SmartNotifier extends ScryptedDeviceBase implements DeviceProvider, MixinProvider, Settings {
    private reidEngine: ReIDEngine;

    constructor(nativeId?: string) {
        super(nativeId);
        this.reidEngine = new ReIDEngine();
    }

    async initialize() {
        await this.reidEngine.initialize();
        log?.i('Smart Notifier initialized with ReID engine');
    }

    async getDevice(nativeId: string): Promise<Device> {
        throw new Error('Smart Notifier does not provide devices');
    }

    async releaseDevice(id: string, nativeId: string): Promise<void> {
        // No devices to release
    }

    async canMixin(type: ScryptedDeviceType, interfaces: string[]): Promise<string[]> {
        if (type === ScryptedDeviceType.Camera && interfaces.includes(ScryptedInterface.ObjectDetection)) {
            return [ScryptedInterface.ObjectDetection, ScryptedInterface.Settings];
        }
        return [];
    }

    async getMixin(mixinDevice: any, mixinDeviceInterfaces: ScryptedInterface[], mixinDeviceState: WritableDeviceState): Promise<any> {
        return new SmartNotifierMixin(mixinDevice, mixinDeviceInterfaces, mixinDeviceState, this.reidEngine, this.nativeId || 'smart-notifier');
    }

    async releaseMixin(id: string, mixinDevice: any): Promise<void> {
        mixinDevice.release?.();
    }

    async getSettings(): Promise<Setting[]> {
        return [
            {
                key: 'debugMode',
                title: 'Debug Mode',
                description: 'Enable debug logging',
                type: 'boolean',
                value: this.storage.getItem('debugMode') || false
            }
        ];
    }

    async putSetting(key: string, value: SettingValue): Promise<void> {
        this.storage.setItem(key, value);

        if (key === 'debugMode') {
            this.reidEngine.setDebugMode(value as boolean);
        }
    }
}

class SmartNotifierMixin extends ScryptedDeviceBase implements ObjectDetection, Settings {
    private device: ScryptedDevice & ObjectDetection;
    private reidEngine: ReIDEngine;
    private cameraId: string;
    private cameraName: string;
    private trackingWindowSeconds: number;
    private enabled: boolean;
    private listener?: EventListenerRegister;

    constructor(
        mixinDevice: ScryptedDevice & ObjectDetection,
        mixinDeviceInterfaces: ScryptedInterface[],
        mixinDeviceState: WritableDeviceState,
        reidEngine: ReIDEngine,
        mixinProviderNativeId: string
    ) {
        super(mixinProviderNativeId);
        this.device = mixinDevice;
        this.reidEngine = reidEngine;
        this.cameraId = mixinDevice.id;
        this.cameraName = mixinDevice.name || 'Unknown Camera';

        // Load saved settings
        this.enabled = this.storage.getItem('enabled') ?? true;
        this.trackingWindowSeconds = this.storage.getItem('trackingWindowSeconds') || TRACKING_WINDOW_DEFAULT;
        this.reidEngine.setTrackingWindow(this.trackingWindowSeconds * 1000);

        // Start listening to detection events if enabled
        if (this.enabled) {
            this.startListening();
        }
    }

    private async startListening() {
        try {
            // Listen for detection events from this camera
            this.listener = await this.device.listen(ScryptedInterface.ObjectDetection, (source, details, data) => {
                this.handleDetectionEvent(details, data).catch(err => {
                    log?.e(`Error handling detection event: ${err}`);
                });
            });

            log?.i(`Smart Notifier listening to ${this.cameraName}`);
        } catch (err) {
            log?.e(`Failed to start listening to ${this.cameraName}: ${err}`);
        }
    }

    private async stopListening() {
        if (this.listener) {
            await this.listener.removeListener();
            this.listener = undefined;
            log?.i(`Smart Notifier stopped listening to ${this.cameraName}`);
        }
    }

    private async handleDetectionEvent(details: EventDetails, data: any) {
        if (!this.enabled) {
            return;
        }

        const eventData = data as ObjectsDetected;
        if (!eventData?.detections?.length) {
            return;
        }

        // Process each detection
        for (const detection of eventData.detections) {
            await this.processDetection(detection, eventData);
        }
    }

    private async processDetection(detection: any, eventData: ObjectsDetected) {
        try {
            // Get detection type (person, face, vehicle, animal)
            const detectionType = detection.className || detection.class || 'unknown';

            // Skip if not a relevant detection type
            if (!['person', 'face', 'vehicle', 'animal'].includes(detectionType)) {
                return;
            }

            // Get the detection snapshot using getRecordingStreamThumbnail
            const camera = systemManager.getDeviceById<VideoRecorder>(this.cameraId);
            if (!camera) {
                log?.e(`Camera not found: ${this.cameraId}`);
                return;
            }

            const timestamp = detection.timestamp || Date.now();
            const thumbnail = await camera.getRecordingStreamThumbnail(timestamp);

            if (!thumbnail) {
                log?.w(`No thumbnail available for ${detectionType} detection`);
                return;
            }

            const imageBuffer = await sdk.mediaManager.convertMediaObjectToBuffer(thumbnail, 'image/jpeg');

            // Process with ReID engine
            const result = await this.reidEngine.processDetection(
                imageBuffer,
                this.cameraId,
                this.cameraName,
                detectionType
            );

            // If new detection, send notification
            if (result.isNew) {
                await this.sendNotification(detection, detectionType, result.firstDetection);
            } else {
                log?.d(`Suppressed ${detectionType} notification (already seen: ${result.personId}`);
            }

        } catch (err) {
            log?.e(`Error processing detection: ${err}`);
        }
    }


    private async sendNotification(
        detection: any,
        detectionType: string,
        firstDetection?: any
    ) {
        try {
            // Build notification title and body
            const title = `${this.capitalize(detectionType)} detected`;
            const body = `${this.capitalize(detectionType)} detected at ${this.cameraName}`;

            // Get all notifier devices from device manager
            const notifierIds = Object.keys(systemManager.getSystemState())
                .filter(id => {
                    const device = systemManager.getDeviceById(id);
                    return device && device.type === ScryptedDeviceType.Notifier;
                });

            if (notifierIds.length === 0) {
                log?.w('No notifier devices available');
                return;
            }

            // Create notification options
            const options: NotifierOptions = {
                body,
                data: {
                    detectionType,
                    cameraId: this.cameraId,
                    cameraName: this.cameraName,
                    timestamp: Date.now()
                }
            };

            // Add snapshot if available
            if (firstDetection?.snapshot) {
                try {
                    // Create media object from snapshot buffer
                    // Create media URL from buffer
                    const mediaUrl = await sdk.mediaManager.createMediaObjectFromUrl(
                        `data:image/jpeg;base64,${firstDetection.snapshot.toString('base64')}`
                    );
                    // Some notifiers support media field
                    (options as any).media = mediaUrl;
                } catch (err) {
                    log?.w(`Failed to attach snapshot: ${err}`);
                }
            }

            // Send notification to all notifiers
            for (const notifierId of notifierIds) {
                try {
                    const notifier = systemManager.getDeviceById<Notifier>(notifierId);
                    if (!notifier) continue;

                    await notifier.sendNotification(title, options);
                    log?.i(`Sent ${detectionType} notification for ${this.cameraName}`);
                } catch (err) {
                    log?.e(`Failed to send via notifier ${notifierId}: ${err}`);
                }
            }

        } catch (err) {
            log?.e(`Failed to send notification: ${err}`);
        }
    }

    private capitalize(str: string): string {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // ObjectDetection interface implementation
    async generateObjectDetections(
        videoFrames: MediaObject | AsyncGenerator<VideoFrame, void, any>,
        session: ObjectDetectionGeneratorSession
    ): Promise<AsyncGenerator<{ __json_copy_serialize_children: true; videoFrame: VideoFrame; detected: ObjectsDetected }, void, any>> {
        // Pass through to original device
        return this.device.generateObjectDetections(videoFrames, session);
    }

    async detectObjects(mediaObject: MediaObject, session?: any): Promise<ObjectsDetected> {
        // Pass through to original device
        return this.device.detectObjects(mediaObject, session);
    }

    async getDetectionModel(settings?: { [key: string]: any }): Promise<ObjectDetectionModel> {
        // Pass through to original device
        return this.device.getDetectionModel(settings);
    }

    // These methods are optional and not needed for passthrough

    // Settings interface
    async getSettings(): Promise<Setting[]> {
        return [
            {
                key: 'enabled',
                title: 'Smart Notifier Enabled',
                description: 'Enable ReID-based notification deduplication',
                type: 'boolean',
                value: this.enabled
            },
            {
                key: 'trackingWindowSeconds',
                title: 'Tracking Window (seconds)',
                description: 'How long to remember a person before sending a new notification',
                type: 'number',
                value: this.trackingWindowSeconds,
                placeholder: TRACKING_WINDOW_DEFAULT.toString()
            }
        ];
    }

    async putSetting(key: string, value: SettingValue): Promise<void> {
        this.storage.setItem(key, value);

        if (key === 'trackingWindowSeconds') {
            this.trackingWindowSeconds = value as number || TRACKING_WINDOW_DEFAULT;
            this.reidEngine.setTrackingWindow(this.trackingWindowSeconds * 1000);
        } else if (key === 'enabled') {
            this.enabled = value as boolean;
            if (this.enabled) {
                this.startListening();
            } else {
                this.stopListening();
            }
        }
    }

    release() {
        this.stopListening();
        log?.i(`Smart Notifier mixin released for camera ${this.cameraId}`);
    }
}

// Create and export the plugin instance
const plugin = new SmartNotifier();
plugin.initialize().catch(err => log?.e(`Failed to initialize Smart Notifier: ${err}`));

export default plugin;