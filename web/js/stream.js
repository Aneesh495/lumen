/**
 * stream.js - Lumen Native Frame Streaming Client
 *
 * Connects to a local Lumen daemon WebSocket or HTTP endpoint to stream
 * hardware-accelerated frames rendered by native Mojo and C kernels.
 */

class LumenStreamClient {
    constructor(url = "ws://localhost:8080/stream") {
        this.url = url;
        this.socket = null;
        this.connected = false;
        this.onFrameCallback = null;
    }

    connect() {
        try {
            this.socket = new WebSocket(this.url);
            this.socket.binaryType = "arraybuffer";

            this.socket.onopen = () => {
                this.connected = true;
                console.log("[LumenStream] Connected to native daemon at " + this.url);
            };

            this.socket.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    if (this.onFrameCallback) {
                        this.onFrameCallback(event.data);
                    }
                }
            };

            this.socket.onclose = () => {
                this.connected = false;
                console.log("[LumenStream] Disconnected from daemon. Reconnecting in 3s...");
                setTimeout(() => this.connect(), 3000);
            };

            this.socket.onerror = () => {
                this.socket.close();
            };
        } catch (err) {
            console.warn("[LumenStream] Daemon not available; running standalone browser client.");
        }
    }

    sendCameraUpdate(cam) {
        if (!this.connected || !this.socket) return;
        const payload = JSON.stringify({
            type: "camera_update",
            eye: cam.eye,
            target: cam.target,
            fov: cam.fov,
        });
        this.socket.send(payload);
    }

    onFrame(cb) {
        this.onFrameCallback = cb;
    }
}

// Global stream client instance
window.LumenStream = new LumenStreamClient();
