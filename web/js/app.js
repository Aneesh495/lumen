/**
 * app.js - Lumen Spatial Radiance Field Interactive Web Viewer
 *
 * Real-time 3D Gaussian Splatting rasterizer, camera orbit controller,
 * and spatial telemetry monitor.
 */

(function () {
    "use strict";

    // Application State
    const state = {
        scene: "cube",
        splatCount: 512,
        renderMode: "radiance",
        fovDeg: 45,
        sigmaRadius: 3.0,
        autoOrbit: true,
        camera: {
            distance: 3.2,
            azimuth: 0.8,
            elevation: 0.35,
            target: [0, 0, 0],
        },
        mouse: {
            down: false,
            lastX: 0,
            lastY: 0,
        },
        telemetry: {
            fps: 60.0,
            latencyMs: 1.0,
            visibleSplats: 0,
            culledSplats: 0,
            touchedTiles: 0,
            earlyTermRays: 0,
        },
    };

    // Gaussian Cloud Storage
    let gaussians = [];

    // DOM Elements
    const canvas = document.getElementById("render-canvas");
    const ctx = canvas.getContext("2d");

    const metricFps = document.getElementById("metric-fps");
    const metricLatency = document.getElementById("metric-latency");
    const metricSplats = document.getElementById("metric-splats");
    const metricTiles = document.getElementById("metric-tiles");

    const sceneSelect = document.getElementById("scene-select");
    const splatDensity = document.getElementById("splat-density");
    const densityVal = document.getElementById("density-val");
    const camFov = document.getElementById("cam-fov");
    const fovVal = document.getElementById("fov-val");
    const gaussianSigma = document.getElementById("gaussian-sigma");
    const sigmaVal = document.getElementById("sigma-val");
    const autoOrbitCheck = document.getElementById("auto-orbit");

    const telCulled = document.getElementById("tel-culled");
    const telVisible = document.getElementById("tel-visible");
    const telTouchedTiles = document.getElementById("tel-touched-tiles");
    const telEarlyTerm = document.getElementById("tel-early-term");

    // Initialize Canvas Resizing
    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
    }
    window.addEventListener("resize", resizeCanvas);

    // Procedural Scene Generators
    function generateCube(count) {
        const list = [];
        const perAxis = Math.max(2, Math.round(Math.cbrt(count)));
        const total = perAxis * perAxis * perAxis;
        const step = perAxis > 1 ? 1.6 / (perAxis - 1) : 0;
        const half = 0.8;
        const splatScale = step * 0.55;

        for (let x = 0; x < perAxis; ++x) {
            const px = -half + x * step;
            for (let y = 0; y < perAxis; ++y) {
                const py = -half + y * step;
                for (let z = 0; z < perAxis; ++z) {
                    const pz = -half + z * step;
                    const r = (px + half) / 1.6;
                    const g = (py + half) / 1.6;
                    const b = (pz + half) / 1.6;

                    list.push({
                        pos: [px, py, pz],
                        scale: [splatScale, splatScale, splatScale],
                        color: [r, g, b],
                        opacity: 0.85,
                    });
                }
            }
        }
        return list;
    }

    function generateSphere(count) {
        const list = [];
        const golden = 1.6180339887;
        const radius = 1.0;
        const splatScale = radius * 0.12;

        for (let i = 0; i < count; ++i) {
            const theta = 2.0 * Math.PI * i / golden;
            const phi = Math.acos(1.0 - 2.0 * (i + 0.5) / count);

            const px = radius * Math.sin(phi) * Math.cos(theta);
            const py = radius * Math.sin(phi) * Math.sin(theta);
            const pz = radius * Math.cos(phi);

            const r = 0.5 + 0.5 * (px / radius);
            const g = 0.5 + 0.5 * (py / radius);
            const b = 0.5 + 0.5 * (pz / radius);

            list.push({
                pos: [px, py, pz],
                scale: [splatScale, splatScale, splatScale],
                color: [r, g, b],
                opacity: 0.9,
            });
        }
        return list;
    }

    function generateTorus(count) {
        const list = [];
        const majorR = 0.9;
        const minorR = 0.35;
        const numMajor = Math.round(Math.sqrt(count * 1.5));
        const numMinor = Math.max(4, Math.round(count / numMajor));

        for (let i = 0; i < numMajor; ++i) {
            const u = (i / numMajor) * 2 * Math.PI;
            for (let j = 0; j < numMinor; ++j) {
                const v = (j / numMinor) * 2 * Math.PI;

                const px = (majorR + minorR * Math.cos(v)) * Math.cos(u);
                const py = minorR * Math.sin(v);
                const pz = (majorR + minorR * Math.cos(v)) * Math.sin(u);

                const r = 0.5 + 0.5 * Math.sin(u);
                const g = 0.5 + 0.5 * Math.cos(v);
                const b = 0.8;

                list.push({
                    pos: [px, py, pz],
                    scale: [0.08, 0.08, 0.08],
                    color: [r, g, b],
                    opacity: 0.85,
                });
            }
        }
        return list;
    }

    function generateHelix(count) {
        const list = [];
        const radius = 0.7;
        const height = 1.8;
        const turns = 3.0;

        for (let i = 0; i < count; ++i) {
            const t = (i / count);
            const angle = t * turns * 2.0 * Math.PI;
            const py = -height * 0.5 + t * height;

            // Strand 1
            const px1 = radius * Math.cos(angle);
            const pz1 = radius * Math.sin(angle);
            list.push({
                pos: [px1, py, pz1],
                scale: [0.07, 0.07, 0.07],
                color: [0.2, 0.8, 0.9],
                opacity: 0.88,
            });

            // Strand 2
            const px2 = radius * Math.cos(angle + Math.PI);
            const pz2 = radius * Math.sin(angle + Math.PI);
            list.push({
                pos: [px2, py, pz2],
                scale: [0.07, 0.07, 0.07],
                color: [0.9, 0.3, 0.5],
                opacity: 0.88,
            });
        }
        return list;
    }

    function rebuildScene() {
        if (state.scene === "cube") {
            gaussians = generateCube(state.splatCount);
        } else if (state.scene === "sphere") {
            gaussians = generateSphere(state.splatCount);
        } else if (state.scene === "torus") {
            gaussians = generateTorus(state.splatCount);
        } else if (state.scene === "helix") {
            gaussians = generateHelix(Math.floor(state.splatCount / 2));
        }
        metricSplats.textContent = gaussians.length;
    }

    // Camera Look-At & Transform
    function computeCameraMatrix() {
        const dist = state.camera.distance;
        const az = state.camera.azimuth;
        const el = state.camera.elevation;

        const eyeX = dist * Math.cos(el) * Math.sin(az);
        const eyeY = dist * Math.sin(el);
        const eyeZ = dist * Math.cos(el) * Math.cos(az);

        // Forward vector (target - eye)
        const fx = -eyeX;
        const fy = -eyeY;
        const fz = -eyeZ;
        const flen = Math.hypot(fx, fy, fz) || 1.0;
        const f = [fx / flen, fy / flen, fz / flen];

        // Right vector (cross f and world up [0, 1, 0])
        const rx = f[2];
        const ry = 0;
        const rz = -f[0];
        const rlen = Math.hypot(rx, rz) || 1.0;
        const r = [rx / rlen, 0, rz / rlen];

        // True up (cross r and f)
        const u = [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ];

        return {
            eye: [eyeX, eyeY, eyeZ],
            r: r,
            u: u,
            f: f,
        };
    }

    // Rendering Pipeline
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsTimer = 0;

    function renderFrame(now) {
        const dt = (now - lastTime) / 1000.0;
        lastTime = now;

        if (state.autoOrbit && !state.mouse.down) {
            state.camera.azimuth += dt * 0.45;
        }

        const tStart = performance.now();
        const w = canvas.width;
        const h = canvas.height;

        if (w === 0 || h === 0) {
            requestAnimationFrame(renderFrame);
            return;
        }

        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, w, h);

        const cam = computeCameraMatrix();
        const fovRad = (state.fovDeg * Math.PI) / 180.0;
        const focal = h / (2.0 * Math.tan(fovRad * 0.5));

        const projected = [];
        let culled = 0;

        // Project 3D Gaussians to screen space
        for (let i = 0; i < gaussians.length; ++i) {
            const g = gaussians[i];
            const dx = g.pos[0] - cam.eye[0];
            const dy = g.pos[1] - cam.eye[1];
            const dz = g.pos[2] - cam.eye[2];

            // Camera coordinate space
            const vx = cam.r[0] * dx + cam.r[1] * dy + cam.r[2] * dz;
            const vy = cam.u[0] * dx + cam.u[1] * dy + cam.u[2] * dz;
            const vz = cam.f[0] * dx + cam.f[1] * dy + cam.f[2] * dz;

            if (vz <= 0.1 || vz >= 50.0) {
                culled++;
                continue;
            }

            const sx = (vx / vz) * focal + w * 0.5;
            const sy = (-vy / vz) * focal + h * 0.5;

            const maxScale = Math.max(g.scale[0], g.scale[1], g.scale[2]);
            const radius = (maxScale * focal / vz) * state.sigmaRadius;

            if (radius < 1.0) continue;
            if (sx + radius < 0 || sx - radius >= w || sy + radius < 0 || sy - radius >= h) {
                culled++;
                continue;
            }

            projected.push({
                sx: sx,
                sy: sy,
                depth: vz,
                radius: radius,
                color: g.color,
                opacity: g.opacity,
            });
        }

        // Sort front-to-back by depth
        projected.sort((a, b) => a.depth - b.depth);

        // Tile Grid Setup
        const tileSize = 32;
        const tilesX = Math.ceil(w / tileSize);
        const tilesY = Math.ceil(h / tileSize);
        const touchedTiles = new Set();

        // Render based on selected visualization mode
        if (state.renderMode === "radiance") {
            for (let i = projected.length - 1; i >= 0; --i) {
                const p = projected[i];
                const grad = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, p.radius);
                const cr = Math.floor(p.color[0] * 255);
                const cg = Math.floor(p.color[1] * 255);
                const cb = Math.floor(p.color[2] * 255);

                grad.addColorStop(0, `rgba(${cr}, ${cg}, ${cb}, ${p.opacity})`);
                grad.addColorStop(0.5, `rgba(${cr}, ${cg}, ${cb}, ${p.opacity * 0.6})`);
                grad.addColorStop(1, `rgba(${cr}, ${cg}, ${cb}, 0)`);

                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(p.sx, p.sy, p.radius, 0, 2 * Math.PI);
                ctx.fill();

                // Compute touched tiles
                const minTx = Math.max(0, Math.floor((p.sx - p.radius) / tileSize));
                const maxTx = Math.min(tilesX - 1, Math.floor((p.sx + p.radius) / tileSize));
                const minTy = Math.max(0, Math.floor((p.sy - p.radius) / tileSize));
                const maxTy = Math.min(tilesY - 1, Math.floor((p.sy + p.radius) / tileSize));

                for (let ty = minTy; ty <= maxTy; ++ty) {
                    for (let tx = minTx; tx <= maxTx; ++tx) {
                        touchedTiles.add(ty * tilesX + tx);
                    }
                }
            }
        } else if (state.renderMode === "depth") {
            for (let i = projected.length - 1; i >= 0; --i) {
                const p = projected[i];
                const normD = Math.max(0, Math.min(1, (p.depth - 1.0) / 4.0));
                const intensity = Math.floor((1.0 - normD) * 255);
                const r = intensity;
                const g = Math.floor(intensity * 0.7);
                const b = Math.floor(255 - intensity * 0.6);

                const grad = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, p.radius);
                grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${p.opacity})`);
                grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(p.sx, p.sy, p.radius, 0, 2 * Math.PI);
                ctx.fill();
            }
        } else if (state.renderMode === "tiles") {
            // Draw tile grid
            ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
            ctx.lineWidth = 1;
            for (let tx = 0; tx <= tilesX; ++tx) {
                ctx.beginPath();
                ctx.moveTo(tx * tileSize, 0);
                ctx.lineTo(tx * tileSize, h);
                ctx.stroke();
            }
            for (let ty = 0; ty <= tilesY; ++ty) {
                ctx.beginPath();
                ctx.moveTo(0, ty * tileSize);
                ctx.lineTo(w, ty * tileSize);
                ctx.stroke();
            }

            // Highlight touched tiles
            for (let i = 0; i < projected.length; ++i) {
                const p = projected[i];
                const minTx = Math.max(0, Math.floor((p.sx - p.radius) / tileSize));
                const maxTx = Math.min(tilesX - 1, Math.floor((p.sx + p.radius) / tileSize));
                const minTy = Math.max(0, Math.floor((p.sy - p.radius) / tileSize));
                const maxTy = Math.min(tilesY - 1, Math.floor((p.sy + p.radius) / tileSize));

                for (let ty = minTy; ty <= maxTy; ++ty) {
                    for (let tx = minTx; tx <= maxTx; ++tx) {
                        touchedTiles.add(ty * tilesX + tx);
                    }
                }
            }

            ctx.fillStyle = "rgba(52, 211, 153, 0.15)";
            touchedTiles.forEach(tileIdx => {
                const tx = tileIdx % tilesX;
                const ty = Math.floor(tileIdx / tilesX);
                ctx.fillRect(tx * tileSize, ty * tileSize, tileSize, tileSize);
            });
        } else if (state.renderMode === "wireframe") {
            ctx.strokeStyle = "rgba(129, 140, 248, 0.7)";
            ctx.lineWidth = 1.5;
            for (let i = 0; i < projected.length; ++i) {
                const p = projected[i];
                ctx.beginPath();
                ctx.ellipse(p.sx, p.sy, p.radius, p.radius * 0.7, 0, 0, 2 * Math.PI);
                ctx.stroke();
            }
        }

        const tEnd = performance.now();
        const latency = tEnd - tStart;

        // Telemetry Update
        frameCount++;
        fpsTimer += dt;
        if (fpsTimer >= 0.5) {
            state.telemetry.fps = (frameCount / fpsTimer).toFixed(1);
            state.telemetry.latencyMs = latency.toFixed(1);
            state.telemetry.visibleSplats = projected.length;
            state.telemetry.culledSplats = culled;
            state.telemetry.touchedTiles = touchedTiles.size || (tilesX * tilesY);

            metricFps.textContent = state.telemetry.fps;
            metricLatency.textContent = state.telemetry.latencyMs + " ms";
            metricTiles.textContent = tilesX * tilesY;

            telCulled.textContent = culled;
            telVisible.textContent = projected.length;
            telTouchedTiles.textContent = state.telemetry.touchedTiles;

            frameCount = 0;
            fpsTimer = 0;
        }

        requestAnimationFrame(renderFrame);
    }

    // Mouse Interaction
    canvas.addEventListener("mousedown", (e) => {
        state.mouse.down = true;
        state.mouse.lastX = e.clientX;
        state.mouse.lastY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
        state.mouse.down = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (!state.mouse.down) return;
        const dx = e.clientX - state.mouse.lastX;
        const dy = e.clientY - state.mouse.lastY;
        state.mouse.lastX = e.clientX;
        state.mouse.lastY = e.clientY;

        state.camera.azimuth += dx * 0.007;
        state.camera.elevation += dy * 0.007;
        state.camera.elevation = Math.max(-1.4, Math.min(1.4, state.camera.elevation));
    });

    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        state.camera.distance += e.deltaY * 0.003;
        state.camera.distance = Math.max(1.0, Math.min(12.0, state.camera.distance));
    }, { passive: false });

    window.addEventListener("keydown", (e) => {
        if (e.code === "Space") {
            e.preventDefault();
            state.autoOrbit = !state.autoOrbit;
            autoOrbitCheck.checked = state.autoOrbit;
        }
    });

    // UI Event Listeners
    sceneSelect.addEventListener("change", (e) => {
        state.scene = e.target.value;
        rebuildScene();
    });

    splatDensity.addEventListener("input", (e) => {
        state.splatCount = parseInt(e.target.value, 10);
        densityVal.textContent = state.splatCount;
        rebuildScene();
    });

    camFov.addEventListener("input", (e) => {
        state.fovDeg = parseInt(e.target.value, 10);
        fovVal.textContent = state.fovDeg + " deg";
    });

    gaussianSigma.addEventListener("input", (e) => {
        state.sigmaRadius = parseFloat(e.target.value);
        sigmaVal.textContent = state.sigmaRadius.toFixed(1);
    });

    autoOrbitCheck.addEventListener("change", (e) => {
        state.autoOrbit = e.target.checked;
    });

    document.querySelectorAll("input[name='render-mode']").forEach(radio => {
        radio.addEventListener("change", (e) => {
            state.renderMode = e.target.value;
        });
    });

    // Startup
    resizeCanvas();
    rebuildScene();
    requestAnimationFrame(renderFrame);
})();
