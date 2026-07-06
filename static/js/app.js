/*
 * AttendAI Vision - static/js/app.js
 *
 * Full frontend controller.
 * Registration capture mirrors the VS Code CLI flow:
 *   FRONT 10 -> immediate backend duplicate check
 *   LEFT 10 -> RIGHT 10 -> UP 10 -> DOWN 10
 *   -> finalize registration
 */

"use strict";

document.addEventListener("DOMContentLoaded", () => {
    initSidebar();
    initToastClose();
    initDashboardStatus();
    initRegistrationForm();
    initCapturePage();
    initLiveAttendance();
    initSettingsPage();
});


/* -------------------------------------------------------------------------- */
/* General helpers                                                            */
/* -------------------------------------------------------------------------- */

function initSidebar() {
    const menuButton = document.getElementById("menu-button");
    const sidebar = document.getElementById("sidebar");
    const mainContent = document.querySelector(".main-content");

    if (!menuButton || !sidebar || !mainContent) return;

    menuButton.addEventListener("click", () => {
        if (window.innerWidth <= 900) {
            sidebar.classList.toggle("open");
        } else {
            sidebar.classList.toggle("collapsed");
            mainContent.classList.toggle("sidebar-collapsed");
        }
    });
}


function initToastClose() {
    document.addEventListener("click", (event) => {
        const closeButton = event.target.closest("[data-toast-close]");
        if (!closeButton) return;

        const toast = closeButton.closest(".toast");
        if (toast) toast.remove();
    });
}


function showToast(message, type = "info", duration = 4500) {
    let container = document.getElementById("toast-container");

    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.className = "toast-container";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const text = document.createElement("span");
    text.textContent = message;

    const close = document.createElement("button");
    close.type = "button";
    close.setAttribute("aria-label", "Close notification");
    close.setAttribute("data-toast-close", "true");
    close.textContent = "×";

    toast.appendChild(text);
    toast.appendChild(close);
    container.appendChild(toast);

    if (duration > 0) {
        window.setTimeout(() => {
            if (toast.isConnected) toast.remove();
        }, duration);
    }
}


async function readJsonResponse(response) {
    try {
        return await response.json();
    } catch {
        return {
            success: false,
            message: `Server returned an invalid response (${response.status}).`,
        };
    }
}


function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}


/* -------------------------------------------------------------------------- */
/* Dashboard                                                                  */
/* -------------------------------------------------------------------------- */

function initDashboardStatus() {
    const modelStatus = document.getElementById("model-status");
    const studentCount = document.getElementById("student-count");
    const presentToday = document.getElementById("present-today");

    if (!modelStatus && !studentCount && !presentToday) return;

    const refresh = async () => {
        try {
            const response = await fetch("/api/system/status", {
                headers: { "Accept": "application/json" },
                cache: "no-store",
            });

            const data = await readJsonResponse(response);
            if (!response.ok) return;

            if (modelStatus) {
                modelStatus.textContent = data.model_ready ? "Ready" : "Not Ready";
                modelStatus.classList.toggle("ready", Boolean(data.model_ready));
            }

            if (studentCount && data.student_count !== undefined) {
                studentCount.textContent = String(data.student_count);
            }

            if (presentToday && data.present_today !== undefined) {
                presentToday.textContent = String(data.present_today);
            }
        } catch {
            // Dashboard polling must never break the page.
        }
    };

    refresh();
    window.setInterval(refresh, 10000);
}


/* -------------------------------------------------------------------------- */
/* Registration form                                                         */
/* -------------------------------------------------------------------------- */

function initRegistrationForm() {
    const form = document.getElementById("student-registration-form");
    if (!form) return;

    const rollInput = document.getElementById("roll-no");
    const nameInput = document.getElementById("student-name");

    // Live roll-number normalization:
    // b230238ee -> B230238EE
    if (rollInput) {
        rollInput.addEventListener("input", () => {
            const cursorPosition = rollInput.selectionStart;

            rollInput.value = rollInput.value.toUpperCase();

            if (cursorPosition !== null) {
                rollInput.setSelectionRange(
                    cursorPosition,
                    cursorPosition
                );
            }
        });
    }

    // Live student-name normalization:
    // gowtham -> Gowtham
    // banothu gowtham -> Banothu Gowtham
    if (nameInput) {
        nameInput.addEventListener("input", () => {
            const cursorPosition = nameInput.selectionStart;

            nameInput.value = nameInput.value
                .toLowerCase()
                .replace(
                    /(^|\s)([a-z])/g,
                    (match, separator, letter) =>
                        separator + letter.toUpperCase()
                );

            if (cursorPosition !== null) {
                nameInput.setSelectionRange(
                    cursorPosition,
                    cursorPosition
                );
            }
        });
    }

    const submitButton =
        form.querySelector('button[type="submit"]') ||
        document.getElementById("register-student-button");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const formData = new FormData(form);

        const payload = {
            roll_no: String(formData.get("roll_no") || "")
                .trim()
                .toUpperCase(),

            name: String(formData.get("name") || "")
                .trim()
                .toLowerCase()
                .replace(
                    /(^|\s)([a-z])/g,
                    (match, separator, letter) =>
                        separator + letter.toUpperCase()
                ),


            department: String(formData.get("department") || "")
                .trim(),
        };

        if (!payload.roll_no || !payload.name || !payload.department) {
            showToast("Enter roll number, name, and department.", "error");
            return;
        }

        const oldText = submitButton ? submitButton.innerHTML : "";

        try {
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = "Validating...";
            }

            const response = await fetch("/api/register/validate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                body: JSON.stringify(payload),
            });

            const data = await readJsonResponse(response);

            if (!response.ok || !data.success) {
                showToast(
                    data.message || "Student registration validation failed.",
                    data.duplicate ? "warning" : "error",
                    6000
                );
                return;
            }

            if (!data.registration_token) {
                showToast(
                    "Registration token was not returned by the server.",
                    "error"
                );
                return;
            }

            window.location.href =
                `/register/capture/${encodeURIComponent(data.registration_token)}`;

        } catch (error) {
            console.error("Registration validation error:", error);
            showToast(
                "Unable to contact the registration server.",
                "error"
            );
        } finally {
            if (submitButton && document.body.contains(submitButton)) {
                submitButton.disabled = false;
                submitButton.innerHTML = oldText;
            }
        }
    });
}


/* -------------------------------------------------------------------------- */
/* Pose-based biometric capture                                               */
/* -------------------------------------------------------------------------- */

function initCapturePage() {
    const sessionData =
        document.getElementById("registration-session-data");

    const video =
        document.getElementById("capture-video");

    const canvas =
        document.getElementById("capture-canvas");

    const startButton =
        document.getElementById("start-camera-button");

    const stopButton =
        document.getElementById("stop-camera-button");

    if (
        !sessionData ||
        !video ||
        !canvas ||
        !startButton ||
        !stopButton
    ) {
        return;
    }

    const cameraStatus =
        document.getElementById("camera-status");

    const captureStatus =
        document.getElementById("capture-status");

    const instruction =
        document.getElementById("capture-instruction");

    const placeholder =
        document.getElementById("camera-placeholder");

    const countElement =
        document.getElementById("capture-count");

    const progressFill =
        document.getElementById("capture-progress-fill");

    const faceGuide =
        document.querySelector(".face-guide");

    const poseNameElement =
        document.getElementById("current-pose-name");

    const poseCountElement =
        document.getElementById("pose-capture-count");

    const countdownElement =
        document.getElementById("capture-countdown");

    const registrationToken =
        sessionData.dataset.registrationToken || "";

    const requiredSampleCount =
        Number.parseInt(
            sessionData.dataset.requiredSampleCount || "50",
            10
        ) || 50;

    const imagesPerPose =
        Number.parseInt(
            sessionData.dataset.imagesPerPose || "10",
            10
        ) || 10;

    const captureIntervalMs =
        Number.parseInt(
            sessionData.dataset.captureIntervalMs || "300",
            10
        ) || 300;

    const countdownSeconds =
        Number.parseInt(
            sessionData.dataset.countdownSeconds || "3",
            10
        ) || 3;

    const POSE_LABELS = {
        front: "FRONT",
        left: "LEFT",
        right: "RIGHT",
        up: "UP",
        down: "DOWN",
    };

    const POSE_INSTRUCTIONS = {
        front: "Look straight at the camera",
        left: "Turn your head to the LEFT",
        right: "Turn your head to the RIGHT",
        up: "Tilt your head UP",
        down: "Tilt your head DOWN",
    };

    let stream = null;
    let running = false;
    let captureLoopRunning = false;
    let finalizeInFlight = false;
    let registrationCompleted = false;
    let registrationRejected = false;

    let currentPose = "front";
    let poseSavedCount = 0;
    let totalSamples = 0;
    let lastSavedAt = 0;

    if (!registrationToken) {
        setCameraStatus("Error");
        setCaptureStatus(
            "Registration session is missing.",
            "error"
        );
        setInstruction(
            "Return to registration and start a new session."
        );
        showToast(
            "Registration session is missing.",
            "error"
        );
        startButton.disabled = true;
        return;
    }

    updateProgress(0);
    updatePoseUI("front", 0);
    setCaptureStatus(
        "Waiting for camera...",
        "ready"
    );

    startButton.addEventListener(
        "click",
        startCamera
    );

    stopButton.addEventListener(
        "click",
        () => stopCamera(true)
    );

    window.addEventListener(
        "beforeunload",
        () => {
            stopMediaTracks();
        }
    );


    async function startCamera() {
        if (
            running ||
            registrationCompleted ||
            registrationRejected
        ) {
            return;
        }

        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {
            setCaptureStatus(
                "Camera access is not supported in this browser.",
                "error"
            );

            showToast(
                "Camera access is not supported in this browser.",
                "error"
            );

            return;
        }

        try {
            setCameraStatus("Starting");

            setCaptureStatus(
                "Requesting camera access...",
                "warning"
            );

            startButton.disabled = true;

            stream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                    },
                    audio: false,
                });

            video.srcObject = stream;
            await video.play();

            running = true;
            stopButton.disabled = false;

            if (placeholder) {
                placeholder.hidden = true;
            }

            setCameraStatus("Active");

            setCaptureStatus(
                "Camera active.",
                "ready"
            );

            const stateOk =
                await syncCaptureState();

            if (!stateOk || !running) {
                return;
            }

            captureLoopRunning = true;

            await runPoseCaptureLoop();

        } catch (error) {
            console.error(
                "Camera startup error:",
                error
            );

            setCameraStatus("Error");

            setCaptureStatus(
                "Unable to start the camera.",
                "error"
            );

            setInstruction(
                "Check browser camera permission and try again."
            );

            showToast(
                "Unable to start camera. Check camera permission.",
                "error"
            );

            stopMediaTracks();

            running = false;
            startButton.disabled = false;
            stopButton.disabled = true;
        }
    }


    async function syncCaptureState() {
        try {
            const response = await fetch(
                `/api/register/capture/${encodeURIComponent(
                    registrationToken
                )}/state`,
                {
                    headers: {
                        "Accept": "application/json"
                    },
                    cache: "no-store",
                }
            );

            const data =
                await readJsonResponse(response);

            if (!response.ok || !data.success) {
                handleSessionOrServerFailure(data);
                return false;
            }

            applyBackendState(data);

            if (data.capture_complete) {
                await finalizeRegistration();
                return false;
            }

            return true;

        } catch (error) {
            console.error(
                "State sync error:",
                error
            );

            setCaptureStatus(
                "Unable to synchronize capture state.",
                "error"
            );

            showToast(
                "Unable to synchronize capture state.",
                "error"
            );

            return false;
        }
    }
    async function runPoseCaptureLoop() {
        try {
            while (
                running &&
                !registrationCompleted &&
                !registrationRejected
            ) {
                if (!currentPose) {
                    await finalizeRegistration();
                    break;
                }

                updatePoseUI(
                    currentPose,
                    poseSavedCount
                );

                // Match the VS Code CLI flow:
                // countdown before each pose stage.
                await runCountdown(currentPose);

                if (!running) {
                    break;
                }

                setCaptureStatus(
                    `Capturing ${POSE_LABELS[currentPose]} samples...`,
                    "capturing"
                );

                while (
                    running &&
                    currentPose &&
                    poseSavedCount < imagesPerPose &&
                    !registrationRejected
                ) {
                    const validation =
                        await validateCurrentFrame();

                    if (
                        !running ||
                        registrationRejected
                    ) {
                        break;
                    }

                    if (!validation) {
                        await sleep(250);
                        continue;
                    }

                    if (!validation.face_valid) {
                        renderValidationFailure(
                            validation.status,
                            validation.message
                        );

                        await sleep(250);
                        continue;
                    }

                    setFaceGuideState("ready");

                    setCaptureStatus(
                        `${POSE_LABELS[currentPose]} face valid — capturing...`,
                        "capturing"
                    );

                    const elapsed =
                        Date.now() - lastSavedAt;

                    if (elapsed < captureIntervalMs) {
                        await sleep(
                            captureIntervalMs - elapsed
                        );
                    }

                    const saveResult =
                        await saveCurrentFrame();

                    if (!saveResult) {
                        await sleep(250);
                        continue;
                    }

                    /*
                     * Critical:
                     * app.py performs duplicate checking
                     * immediately after front_10.jpg.
                     */
                    if (
                        saveResult.status ===
                        "duplicate_face"
                    ) {
                        rejectDuplicate(saveResult);
                        break;
                    }

                    if (!saveResult.success) {
                        if (
                            saveResult.status ===
                            "invalid_session"
                        ) {
                            handleSessionOrServerFailure(
                                saveResult
                            );
                            break;
                        }

                        setCaptureStatus(
                            saveResult.message ||
                                "Sample could not be saved.",
                            "error"
                        );

                        await sleep(350);
                        continue;
                    }

                    /*
                     * Always trust backend counts.
                     * Never increment locally.
                     */
                    applyBackendState(saveResult);

                    if (saveResult.saved) {
                        lastSavedAt = Date.now();
                    }

                    if (saveResult.capture_complete) {
                        await finalizeRegistration();
                        break;
                    }

                    /*
                     * After sample 10, backend advances:
                     * front -> left -> right -> up -> down
                     */
                    if (
                        saveResult.current_pose &&
                        saveResult.current_pose !== currentPose
                    ) {
                        currentPose =
                            saveResult.current_pose;

                        poseSavedCount =
                            Number(
                                saveResult.pose_saved_count || 0
                            );

                        updatePoseUI(
                            currentPose,
                            poseSavedCount
                        );

                        break;
                    }

                    await sleep(30);
                }
            }
        } catch (error) {
            console.error(
                "Capture loop error:",
                error
            );

            setCaptureStatus(
                "Capture process stopped unexpectedly.",
                "error"
            );

            setInstruction(
                "Restart the camera or begin a new registration."
            );

            showToast(
                "Capture process stopped unexpectedly.",
                "error"
            );
        } finally {
            captureLoopRunning = false;
        }
    }


    async function runCountdown(pose) {
        const label =
            POSE_LABELS[pose] ||
            String(pose).toUpperCase();

        const poseInstruction =
            POSE_INSTRUCTIONS[pose] ||
            "Position your face";

        setFaceGuideState("warning");

        for (
            let value = countdownSeconds;
            value >= 1;
            value -= 1
        ) {
            if (!running) {
                return;
            }

            setCaptureStatus(
                `${label} pose starts in ${value}...`,
                "warning"
            );

            setInstruction(
                `${poseInstruction}. Hold steady. ${value}`
            );

            if (countdownElement) {
                countdownElement.textContent =
                    String(value);

                countdownElement.hidden = false;
            }

            await sleep(1000);
        }

        if (countdownElement) {
            countdownElement.hidden = true;
            countdownElement.textContent = "";
        }

        setInstruction(
            `${poseInstruction}. Hold steady while samples are captured.`
        );
    }


    async function validateCurrentFrame() {
        const blob =
            await captureVideoBlob();

        if (!blob) {
            return null;
        }

        const body =
            new FormData();

        body.append(
            "frame",
            blob,
            "frame.jpg"
        );

        try {
            const response = await fetch(
                `/api/register/capture/${encodeURIComponent(
                    registrationToken
                )}/validate-frame`,
                {
                    method: "POST",
                    body,
                    headers: {
                        "Accept": "application/json"
                    },
                }
            );

            const data =
                await readJsonResponse(response);

            if (
                data.status ===
                "invalid_session"
            ) {
                handleSessionOrServerFailure(data);
                return null;
            }

            if (
                data.current_pose !==
                undefined
            ) {
                applyBackendState(data);
            }

            return data;

        } catch (error) {
            console.error(
                "Frame validation error:",
                error
            );

            setCaptureStatus(
                "Temporary camera validation error.",
                "warning"
            );

            return null;
        }
    }


    async function saveCurrentFrame() {
        const blob =
            await captureVideoBlob();

        if (!blob) {
            return null;
        }

        const body =
            new FormData();

        body.append(
            "frame",
            blob,
            "sample.jpg"
        );

        try {
            const response = await fetch(
                `/api/register/capture/${encodeURIComponent(
                    registrationToken
                )}/save-sample`,
                {
                    method: "POST",
                    body,
                    headers: {
                        "Accept": "application/json"
                    },
                }
            );

            return await readJsonResponse(
                response
            );

        } catch (error) {
            console.error(
                "Sample save error:",
                error
            );

            return {
                success: false,
                status: "network_error",
                message:
                    "Unable to save sample because of a network error.",
            };
        }
    }


    async function captureVideoBlob() {
        if (
            !running ||
            video.readyState <
                HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
            return null;
        }

        const width =
            video.videoWidth;

        const height =
            video.videoHeight;

        if (!width || !height) {
            return null;
        }

        canvas.width = width;
        canvas.height = height;

        const context =
            canvas.getContext(
                "2d",
                { alpha: false }
            );

        if (!context) {
            return null;
        }

        context.drawImage(
            video,
            0,
            0,
            width,
            height
        );

        return await new Promise(
            (resolve) => {
                canvas.toBlob(
                    (blob) => resolve(blob),
                    "image/jpeg",
                    0.92
                );
            }
        );
    }


    function applyBackendState(data) {
        if (
            data.saved_count !==
            undefined
        ) {
            totalSamples =
                Number(data.saved_count) || 0;

            updateProgress(
                totalSamples
            );
        }

        if (
            data.current_pose !==
            undefined
        ) {
            currentPose =
                data.current_pose || null;
        }

        if (
            data.pose_saved_count !==
            undefined
        ) {
            poseSavedCount =
                Number(
                    data.pose_saved_count
                ) || 0;
        }

        updatePoseUI(
            currentPose,
            poseSavedCount
        );
    }


    function updatePoseUI(
        pose,
        savedCount
    ) {
        const label = pose
            ? (
                POSE_LABELS[pose] ||
                pose.toUpperCase()
            )
            : "COMPLETE";

        const poseInstruction = pose
            ? (
                POSE_INSTRUCTIONS[pose] ||
                "Position your face"
            )
            : "All pose samples captured";

        if (poseNameElement) {
            poseNameElement.textContent =
                label;
        }

        if (poseCountElement) {
            poseCountElement.textContent =
                pose
                    ? `${savedCount}/${imagesPerPose}`
                    : `${imagesPerPose}/${imagesPerPose}`;
        }

        if (pose && running) {
            setInstruction(
                `${poseInstruction}. ${savedCount}/${imagesPerPose} samples captured.`
            );
        }
    }


    function updateProgress(count) {
        const safeCount =
            Math.max(
                0,
                Math.min(
                    requiredSampleCount,
                    Number(count) || 0
                )
            );

        if (countElement) {
            countElement.textContent =
                String(safeCount);
        }

        if (progressFill) {
            const percentage =
                (
                    safeCount /
                    requiredSampleCount
                ) * 100;

            progressFill.style.width =
                `${Math.min(
                    100,
                    percentage
                )}%`;
        }
    }


    function renderValidationFailure(
        status,
        message
    ) {
        const mapping = {
            no_face: [
                "No face detected.",
                "Position one face inside the guide."
            ],

            multiple_faces: [
                "Multiple faces detected.",
                "Only one person should be visible."
            ],

            face_too_small: [
                "Face is too far away.",
                "Move closer to the camera."
            ],

            face_too_large: [
                "Face is too close.",
                "Move farther from the camera."
            ],

            invalid_frame: [
                "Camera frame is invalid.",
                "Hold steady and try again."
            ],
        };

        const [
            statusText,
            instructionText
        ] =
            mapping[status] ||
            [
                message ||
                    "Face not ready.",
                "Adjust your position."
            ];

        setFaceGuideState(
            "warning"
        );

        setCaptureStatus(
            statusText,
            "warning"
        );

        setInstruction(
            instructionText
        );
    }




    function rejectDuplicate(data) {
        registrationRejected = true;
        running = false;

        stopMediaTracks();

        setCameraStatus("Rejected");

        setFaceGuideState(
            "error"
        );

        setCaptureStatus(
            "Duplicate face detected.",
            "rejected"
        );

        setInstruction(
            data.message ||
            "This face is already registered. Registration has been stopped."
        );

        startButton.disabled = true;
        stopButton.disabled = true;

        showToast(
            data.message ||
            "This face is already registered.",
            "error",
            10000
        );
    }


    function showClassicPipelinePanel() {
        const panel =
            document.getElementById("registration-processing-panel");

        if (panel) {
            panel.hidden = false;
        }
    }


    function updateClassicPipeline(status, message, progress, error = null) {
        const title =
            document.getElementById("processing-title");

        const messageElement =
            document.getElementById("processing-message");

        const progressFill =
            document.getElementById("processing-progress-fill");

        const progressText =
            document.getElementById("processing-progress-text");

        const note =
            document.getElementById("processing-note");

        const retryButton =
            document.getElementById("processing-retry-button");

        const liveButton =
            document.getElementById("processing-live-button");

        showClassicPipelinePanel();

        if (progressFill) {
            progressFill.style.width =
                `${Math.max(0, Math.min(100, Number(progress) || 0))}%`;
        }

        if (progressText) {
            progressText.textContent =
                `${Math.max(0, Math.min(100, Number(progress) || 0))}%`;
        }

        if (messageElement) {
            messageElement.textContent =
                error || message || "Processing registration...";
        }

        if (title) {
            title.textContent =
                status === "ready"
                    ? "Registration Completed"
                    : status === "failed"
                        ? "Registration Failed"
                        : "Processing Registration";
        }

        if (note) {
            note.textContent =
                status === "ready"
                    ? "Student biometric profile is ready for live attendance."
                    : status === "failed"
                        ? "The background registration process could not be completed."
                        : "Please do not close this page.";
        }

        if (retryButton) {
            retryButton.hidden =
                status !== "failed";
        }

        if (liveButton) {
            liveButton.hidden =
                status !== "ready";
        }
    }


    async function pollRegistrationPipeline(jobId) {
        while (true) {
            const response = await fetch(
                `/api/register/pipeline/${encodeURIComponent(jobId)}/status`,
                {
                    method: "GET",
                    headers: {
                        "Accept": "application/json"
                    },
                    cache: "no-store"
                }
            );

            const data =
                await readJsonResponse(response);

            if (!response.ok || !data.success) {
                throw new Error(
                    data.detail ||
                    data.message ||
                    "Unable to read registration progress."
                );
            }

            updateClassicPipeline(
                data.status,
                data.message,
                data.progress,
                data.error
            );

            if (data.status === "ready") {
                setCameraStatus("Ready");
                setFaceGuideState("ready");

                setCaptureStatus(
                    "Registration completed.",
                    "registered"
                );

                showToast(
                    "Student is ready for live attendance.",
                    "success",
                    7000
                );

                return;
            }

            if (data.status === "failed") {
                setCameraStatus("Failed");
                setFaceGuideState("error");

                setCaptureStatus(
                    "Registration processing failed.",
                    "rejected"
                );

                showToast(
                    data.error ||
                    "Registration processing failed.",
                    "error",
                    9000
                );

                return;
            }

            await new Promise(
                (resolve) => window.setTimeout(resolve, 1200)
            );
        }
    }


    async function retryRegistrationPipeline(jobId) {
        const response = await fetch(
            `/api/register/pipeline/${encodeURIComponent(jobId)}/retry`,
            {
                method: "POST",
                headers: {
                    "Accept": "application/json"
                }
            }
        );

        const data =
            await readJsonResponse(response);

        if (!response.ok || !data.success) {
            throw new Error(
                data.detail ||
                data.message ||
                "Unable to retry registration processing."
            );
        }

        updateClassicPipeline(
            "queued",
            data.message || "Retry started.",
            5
        );

        await pollRegistrationPipeline(jobId);
    }


    async function finalizeRegistration() {
        if (
            finalizeInFlight ||
            registrationCompleted ||
            registrationRejected
        ) {
            return;
        }

        finalizeInFlight = true;
        running = false;

        setCameraStatus("Finalizing");
        setFaceGuideState("warning");

        setCaptureStatus(
            "Saving registration...",
            "finalizing"
        );

        setInstruction(
            "Please wait while registration is finalized."
        );

        try {
            const response = await fetch(
                `/api/register/capture/${encodeURIComponent(
                    registrationToken
                )}/finalize`,
                {
                    method: "POST",
                    headers: {
                        "Accept": "application/json"
                    }
                }
            );

            const data =
                await readJsonResponse(response);

            if (data.status === "duplicate_face") {
                rejectDuplicate(data);
                return;
            }

            if (
                !response.ok ||
                !data.success ||
                !data.saved
            ) {
                throw new Error(
                    data.message ||
                    "The server could not finalize registration."
                );
            }

            const jobId =
                data.pipeline_job_id;

            if (!jobId) {
                throw new Error(
                    "Registration started but no pipeline job id was returned."
                );
            }

            registrationCompleted = true;
            stopMediaTracks();

            startButton.disabled = true;
            stopButton.disabled = true;

            setCameraStatus("Processing");
            setFaceGuideState("warning");

            setCaptureStatus(
                "Processing registration...",
                "finalizing"
            );

            setInstruction(
                "The background registration pipeline is running."
            );

            updateClassicPipeline(
                "queued",
                "Preparing registration...",
                5
            );

            const retryButton =
                document.getElementById(
                    "processing-retry-button"
                );

            if (retryButton) {
                retryButton.onclick = async () => {
                    retryButton.disabled = true;

                    try {
                        await retryRegistrationPipeline(jobId);
                    } catch (error) {
                        console.error(
                            "Pipeline retry error:",
                            error
                        );

                        showToast(
                            error.message ||
                            "Unable to retry registration processing.",
                            "error",
                            8000
                        );
                    } finally {
                        retryButton.disabled = false;
                    }
                };
            }

            await pollRegistrationPipeline(jobId);

        } catch (error) {
            console.error(
                "Finalize/pipeline error:",
                error
            );

            setCameraStatus("Error");
            setFaceGuideState("error");

            setCaptureStatus(
                "Registration could not be completed.",
                "rejected"
            );

            setInstruction(
                error.message ||
                "Unable to contact the server."
            );

            showToast(
                error.message ||
                "Unable to finalize registration.",
                "error",
                8000
            );

        } finally {
            finalizeInFlight = false;
        }
    }


    function handleSessionOrServerFailure(
        data
    ) {
        running = false;

        stopMediaTracks();

        setCameraStatus(
            "Error"
        );

        setFaceGuideState(
            "error"
        );

        setCaptureStatus(
            data.message ||
            "Registration session is invalid or expired.",
            "error"
        );

        setInstruction(
            "Return to registration and begin a new session."
        );

        startButton.disabled = true;
        stopButton.disabled = true;

        showToast(
            data.message ||
            "Registration session is invalid or expired.",
            "error",
            7000
        );
    }


    function stopCamera(
        userInitiated = false
    ) {
        running = false;
        captureLoopRunning = false;

        stopMediaTracks();

        /*
         * Never overwrite a completed
         * registration success state.
         */
        if (registrationCompleted) {
            setCameraStatus(
                "Complete"
            );

            setCaptureStatus(
                "Registration complete.",
                "registered"
            );

            return;
        }

        /*
         * Never overwrite duplicate rejection.
         */
        if (registrationRejected) {
            setCameraStatus(
                "Rejected"
            );

            setCaptureStatus(
                "Duplicate face detected.",
                "rejected"
            );

            return;
        }

        setCameraStatus(
            "Stopped"
        );

        setFaceGuideState(
            ""
        );

        setCaptureStatus(
            "Camera stopped.",
            "warning"
        );

        if (userInitiated) {
            setInstruction(
                "Start the camera to continue biometric enrollment."
            );
        } else {
            setInstruction(
                "Start the camera to continue biometric enrollment."
            );
        }

        startButton.disabled = false;
        stopButton.disabled = true;

        if (placeholder) {
            placeholder.hidden = false;
        }
    }


    function stopMediaTracks() {
        if (stream) {
            for (
                const track of stream.getTracks()
            ) {
                track.stop();
            }

            stream = null;
        }

        if (video.srcObject) {
            video.srcObject = null;
        }
    }


    function setCameraStatus(
        text
    ) {
        if (cameraStatus) {
            cameraStatus.textContent =
                text;
        }
    }


    function setCaptureStatus(
        text,
        state = ""
    ) {
        if (!captureStatus) {
            return;
        }

        captureStatus.textContent =
            text;

        const states = [
            "ready",
            "warning",
            "error",
            "capturing",
            "complete",
            "finalizing",
            "registered",
            "rejected",
        ];

        for (
            const item of states
        ) {
            captureStatus.classList.remove(
                item
            );
        }

        if (state) {
            captureStatus.classList.add(
                state
            );
        }
    }


    function setInstruction(
        text
    ) {
        if (instruction) {
            instruction.textContent =
                text;
        }
    }


    function setFaceGuideState(
        state
    ) {
        if (!faceGuide) {
            return;
        }

        const states = [
            "ready",
            "warning",
            "error",
            "capturing",
            "complete",
            "finalizing",
            "registered",
            "rejected",
        ];

        for (
            const item of states
        ) {
            faceGuide.classList.remove(
                item
            );
        }

        if (state) {
            faceGuide.classList.add(
                state
            );
        }
    }
}



/* -------------------------------------------------------------------------- */
/* Live Attendance                                                            */
/* -------------------------------------------------------------------------- */

function initLiveAttendance() {
    const pageData = document.getElementById("attendance-page-data");
    if (!pageData) return;

    const video = document.getElementById("attendance-video");
    const canvas = document.getElementById("attendance-canvas");

    const startButton = document.getElementById(
        "start-attendance-camera-button"
    );

    const stopButton = document.getElementById(
        "stop-attendance-camera-button"
    );

    const cameraStatus = document.getElementById(
        "attendance-camera-status"
    );

    const statusElement = document.getElementById(
        "attendance-status"
    );

    const placeholder = document.getElementById(
        "attendance-camera-placeholder"
    );

    const studentAvatar = document.getElementById(
        "attendance-student-avatar"
    );

    const studentName = document.getElementById(
        "attendance-student-name"
    );

    const studentRoll = document.getElementById(
        "attendance-student-roll"
    );

    const recognitionResult = document.getElementById(
        "attendance-recognition-result"
    );

    const confidenceElement = document.getElementById(
        "attendance-confidence"
    );

    const livenessElement = document.getElementById(
        "attendance-liveness"
    );

    const attendanceResult = document.getElementById(
        "attendance-mark-result"
    );

    const instructionElement = document.getElementById(
        "attendance-instruction"
    );

    if (
        !video ||
        !canvas ||
        !startButton ||
        !stopButton ||
        !cameraStatus ||
        !statusElement
    ) {
        return;
    }

    const modelReady =
        pageData.dataset.modelReady === "true";

    const PROCESS_INTERVAL_MS = 700;
    const SUCCESS_HOLD_MS = 2500;

    let mediaStream = null;
    let processingTimer = null;
    let requestInFlight = false;
    let cameraRunning = false;
    let successHoldUntil = 0;


    function setCameraState(state, message) {
        cameraStatus.textContent = message;

        cameraStatus.classList.remove(
            "ready",
            "warning",
            "error",
            "capturing",
            "complete",
            "registered",
            "rejected"
        );

        if (state) {
            cameraStatus.classList.add(state);
        }
    }


    function setStatus(message) {
        statusElement.textContent = message;
    }


    function setInstruction(message) {
        if (instructionElement) {
            instructionElement.textContent = message;
        }
    }


    function formatPercent(value) {
        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "--%";
        }

        return `${(number * 100).toFixed(1)}%`;
    }


    function resetRecognitionPanel() {
        if (studentAvatar) {
            studentAvatar.textContent = "?";
        }

        if (studentName) {
            studentName.textContent = "Waiting for recognition";
        }

        if (studentRoll) {
            studentRoll.textContent = "No student detected";
        }

        if (recognitionResult) {
            recognitionResult.textContent = "Waiting";
        }

        if (confidenceElement) {
            confidenceElement.textContent = "--%";
        }

        if (livenessElement) {
            livenessElement.textContent = "--%";
        }

        if (attendanceResult) {
            attendanceResult.textContent = "Not marked";
        }

        setInstruction(
            "Position exactly one face inside the guide."
        );
    }


    function clearRecognitionResult({
        name = "Waiting for recognition",
        roll = "No student detected",
        recognition = "Waiting",
        confidence = "--%",
        liveness = "--%",
        attendance = "Not marked",
        avatar = "?"
    } = {}) {
        if (studentAvatar) {
            studentAvatar.textContent = avatar;
        }

        if (studentName) {
            studentName.textContent = name;
        }

        if (studentRoll) {
            studentRoll.textContent = roll;
        }

        if (recognitionResult) {
            recognitionResult.textContent = recognition;
        }

        if (confidenceElement) {
            confidenceElement.textContent = confidence;
        }

        if (livenessElement) {
            livenessElement.textContent = liveness;
        }

        if (attendanceResult) {
            attendanceResult.textContent = attendance;
        }
    }


    function showRecognizedStudent(data) {
        const name = data.name || "Recognized Student";
        const rollNo = data.roll_no || "--";

        if (studentAvatar) {
            studentAvatar.textContent =
                name.charAt(0).toUpperCase() || "?";
        }

        if (studentName) {
            studentName.textContent = name;
        }

        if (studentRoll) {
            studentRoll.textContent = rollNo;
        }

        if (recognitionResult) {
            recognitionResult.textContent = "Recognized";
        }

        if (confidenceElement) {
            confidenceElement.textContent =
                formatPercent(data.confidence);
        }

        if (livenessElement) {
            livenessElement.textContent =
                formatPercent(data.liveness_score);
        }
    }


    function handleAttendanceResult(data) {
        const status = data.status;

        if (status === "marked") {
            showRecognizedStudent(data);

            if (attendanceResult) {
                attendanceResult.textContent = "Marked Present";
            }

            setCameraState(
                "registered",
                "Attendance Marked"
            );

            setStatus(
                `${data.name} marked present`
            );

            setInstruction(
                "Attendance marked successfully."
            );

            successHoldUntil =
                Date.now() + SUCCESS_HOLD_MS;

            return;
        }


        if (status === "already_marked") {
            showRecognizedStudent(data);

            if (attendanceResult) {
                attendanceResult.textContent =
                    "Already Marked Today";
            }

            setCameraState(
                "ready",
                "Attendance Already Marked"
            );

            setStatus(
                `${data.name} attendance already marked today`
            );

            setInstruction(
                "Attendance has already been recorded for today."
            );

            successHoldUntil =
                Date.now() + SUCCESS_HOLD_MS;

            return;
        }


        if (status === "unknown") {
            if (studentAvatar) {
                studentAvatar.textContent = "?";
            }

            if (studentName) {
                studentName.textContent = "Unknown Face";
            }

            if (studentRoll) {
                studentRoll.textContent =
                    "Not registered";
            }

            if (recognitionResult) {
                recognitionResult.textContent =
                    "Unknown";
            }

            if (confidenceElement) {
                confidenceElement.textContent =
                    formatPercent(data.confidence);
            }

            if (livenessElement) {
                livenessElement.textContent =
                    formatPercent(data.liveness_score);
            }

            if (attendanceResult) {
                attendanceResult.textContent =
                    "Not marked";
            }

            setCameraState(
                "warning",
                "Unknown"
            );

            setStatus(
                "Face not recognized"
            );

            setInstruction(
                "This face is not recognized as a registered student."
            );

            return;
        }


        if (status === "spoof_detected") {
            clearRecognitionResult({
                name: "Spoof Detected",
                roll: "Live face required",
                recognition: "Rejected",
                confidence: "--%",
                liveness: formatPercent(data.liveness_score),
                attendance: "Not marked",
                avatar: "!"
            });

            setCameraState(
                "rejected",
                "Spoof Rejected"
            );

            setStatus(
                "Spoof detected"
            );

            setInstruction(
                "A live face is required for attendance."
            );

            return;
        }


        if (status === "no_face") {
            clearRecognitionResult();

            setCameraState(
                "warning",
                "Searching"
            );

            setStatus(
                "No face detected"
            );

            setInstruction(
                "Position one face inside the guide."
            );

            return;
        }


        if (status === "multiple_faces") {
            clearRecognitionResult({
                name: "Multiple Faces",
                roll: "Only one person allowed",
                recognition: "Rejected",
                confidence: "--%",
                liveness: "--%",
                attendance: "Not marked",
                avatar: "!"
            });

            setCameraState(
                "warning",
                "Multiple Faces"
            );

            setStatus(
                "Multiple faces detected"
            );

            setInstruction(
                "Only one person may appear in the camera."
            );

            return;
        }


        if (status === "face_too_small") {
            clearRecognitionResult({
                name: "Move Closer",
                roll: "Face is too far away",
                recognition: "Waiting"
            });

            setCameraState(
                "warning",
                "Move Closer"
            );

            setStatus(
                "Face too small"
            );

            setInstruction(
                "Move closer to the camera."
            );
        if (status === "face_too_large") {
            clearRecognitionResult({
                name: "Move Back",
                roll: "Face is too close",
                recognition: "Waiting"
            });

            setCameraState(
                "warning",
                "Move Back"
            );

            setStatus(
                "Face too large"
            );

            setInstruction(
                "Move farther from the camera."
            );

            return;
        }

            return;
        }
        

        if (status === "student_not_found") {
            setCameraState(
                "error",
                "Database Error"
            );

            setStatus(
                "Student record not found"
            );

            setInstruction(
                data.message ||
                "Recognized identity is missing from the student database."
            );

            return;
        }


        setCameraState(
            "warning",
            "Checking"
        );

        setStatus(
            data.message ||
            "Processing attendance..."
        );
    }


    async function captureFrameBlob() {
        if (
            !cameraRunning ||
            video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
            return null;
        }

        const width = video.videoWidth;
        const height = video.videoHeight;

        if (!width || !height) {
            return null;
        }

        canvas.width = width;
        canvas.height = height;

        const context = canvas.getContext("2d");

        context.drawImage(
            video,
            0,
            0,
            width,
            height
        );

        return await new Promise((resolve) => {
            canvas.toBlob(
                resolve,
                "image/jpeg",
                0.88
            );
        });
    }


    async function processAttendanceFrame() {
        if (
            !cameraRunning ||
            requestInFlight
        ) {
            return;
        }

        if (Date.now() < successHoldUntil) {
            return;
        }

        const blob = await captureFrameBlob();

        if (!blob) {
            return;
        }

        requestInFlight = true;

        try {
            const formData = new FormData();

            formData.append(
                "frame",
                blob,
                "attendance-frame.jpg"
            );

            const response = await fetch(
                "/api/attendance/process-frame",
                {
                    method: "POST",
                    body: formData
                }
            );

            let data = {};

            try {
                data = await response.json();
            } catch (_) {
                data = {};
            }

            if (!response.ok) {
                throw new Error(
                    data.detail ||
                    data.message ||
                    "Attendance processing failed."
                );
            }

            handleAttendanceResult(data);

        } catch (error) {
            console.error(
                "Attendance frame processing error:",
                error
            );

            setCameraState(
                "error",
                "Processing Error"
            );

            setStatus(
                "Attendance processing failed"
            );

            setInstruction(
                error.message ||
                "Unable to process the camera frame."
            );

        } finally {
            requestInFlight = false;
        }
    }


    async function startCamera() {
        if (!modelReady) {
            showToast(
                "Recognition model is not ready.",
                "error"
            );
            return;
        }

        if (cameraRunning) {
            return;
        }

        try {
            mediaStream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "user",
                        width: {
                            ideal: 1280
                        },
                        height: {
                            ideal: 720
                        }
                    },
                    audio: false
                });

            video.srcObject = mediaStream;

            await video.play();

            cameraRunning = true;
            successHoldUntil = 0;

            if (placeholder) {
                placeholder.hidden = true;
            }

            startButton.disabled = true;
            stopButton.disabled = false;

            setCameraState(
                "ready",
                "Camera Active"
            );

            setStatus(
                "Checking face..."
            );

            setInstruction(
                "Position exactly one face inside the guide."
            );

            processingTimer = window.setInterval(
                processAttendanceFrame,
                PROCESS_INTERVAL_MS
            );

        } catch (error) {
            console.error(
                "Unable to start attendance camera:",
                error
            );

            setCameraState(
                "error",
                "Camera Error"
            );

            setStatus(
                "Unable to start camera"
            );

            setInstruction(
                "Allow camera permission and try again."
            );

            showToast(
                "Unable to access the camera.",
                "error"
            );
        }
    }


    function stopCamera() {
        cameraRunning = false;
        successHoldUntil = 0;

        if (processingTimer !== null) {
            window.clearInterval(processingTimer);
            processingTimer = null;
        }

        if (mediaStream) {
            mediaStream
                .getTracks()
                .forEach((track) => track.stop());

            mediaStream = null;
        }

        video.srcObject = null;

        if (placeholder) {
            placeholder.hidden = false;
        }

        startButton.disabled = !modelReady;
        stopButton.disabled = true;

        setCameraState(
            "",
            "Stopped"
        );

        setStatus(
            "Start the camera to begin attendance recognition."
        );

        resetRecognitionPanel();
    }


    startButton.addEventListener(
        "click",
        startCamera
    );

    stopButton.addEventListener(
        "click",
        stopCamera
    );

    window.addEventListener(
        "beforeunload",
        () => {
            if (processingTimer !== null) {
                window.clearInterval(processingTimer);
            }

            if (mediaStream) {
                mediaStream
                    .getTracks()
                    .forEach((track) => track.stop());
            }
        }
    );

    resetRecognitionPanel();
}

/* -------------------------------------------------------------------------- */
/* Attendance Date Filter                                                     */
/* -------------------------------------------------------------------------- */

function initAttendanceDateFilter() {
    const dateSelect =
        document.getElementById("attendance-date-sort");

    const tableBody =
        document.getElementById("attendance-table-body");

    if (!dateSelect || !tableBody) {
        return;
    }

    dateSelect.addEventListener("change", () => {
        const selectedDate = dateSelect.value;

        const rows = tableBody.querySelectorAll(
            "tr[data-attendance-date]"
        );

        rows.forEach((row) => {
            const rowDate =
                row.dataset.attendanceDate;

            if (
                selectedDate === "all" ||
                rowDate === selectedDate
            ) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        });
    });
}


/* Initialize Attendance Date Filter */
document.addEventListener(
    "DOMContentLoaded",
    initAttendanceDateFilter
);




/* Student lifecycle: edit, activate/deactivate, delete, re-enroll */
document.addEventListener("DOMContentLoaded", () => {
    document.addEventListener("click", async (event) => {
        const edit = event.target.closest("[data-student-edit]");
        if (edit) {
            const name = prompt("Student name:", edit.dataset.name || "");
            if (name === null) return;
            const department = prompt("Department:", edit.dataset.department || "");
            if (department === null) return;
            const r = await fetch(`/api/students/${encodeURIComponent(edit.dataset.roll)}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, department }),
            });
            const d = await readJsonResponse(r);
            if (!r.ok) return showToast(d.detail || "Update failed.", "error");
            location.reload();
            return;
        }

        const action = event.target.closest("[data-student-action]");
        if (!action) return;
        const roll = action.dataset.roll;
        const kind = action.dataset.studentAction;

        if (kind === "delete" && !confirm(`Delete ${roll} and its biometric data?`)) return;
        if (kind === "reenroll" && !confirm(`Replace face enrollment for ${roll}?`)) return;

        const method = kind === "delete" ? "DELETE" : "POST";
        const url = kind === "delete"
            ? `/api/students/${encodeURIComponent(roll)}`
            : `/api/students/${encodeURIComponent(roll)}/${kind}`;

        const r = await fetch(url, { method });
        const d = await readJsonResponse(r);
        if (!r.ok) return showToast(d.detail || d.message || "Action failed.", "error");

        if (kind === "reenroll" && d.capture_url) {
            window.location.href = d.capture_url;
            return;
        }
        location.reload();
    });
});
/* -------------------------------------------------------------------------- */
/* Settings Page                                                              */
/* -------------------------------------------------------------------------- */
function initSettingsPage() {
    const form =
        document.getElementById("settings-form");

    if (!form) {
        return;
    }

    const recognitionInput =
        document.getElementById(
            "recognition-confidence-threshold"
        );

    const livenessInput =
        document.getElementById(
            "liveness-score-threshold"
        );

    const arcfaceInput =
        document.getElementById(
            "arcface-verification-threshold"
        );

    const attendanceInput =
        document.getElementById(
            "attendance-once-per-day"
        );

    const cameraInput =
        document.getElementById("camera-index");


    const recognitionValue =
        document.getElementById(
            "recognition-threshold-value"
        );

    const livenessValue =
        document.getElementById(
            "liveness-threshold-value"
        );

    const arcfaceValue =
        document.getElementById(
            "arcface-verification-threshold-value"
        );

    const saveButton =
        document.getElementById(
            "save-settings-button"
        );


    function updateThresholdLabels() {
        if (
            recognitionInput &&
            recognitionValue
        ) {
            recognitionValue.textContent =
                `${Math.round(
                    Number(recognitionInput.value) * 100
                )}%`;
        }

        if (
            livenessInput &&
            livenessValue
        ) {
            livenessValue.textContent =
                `${Math.round(
                    Number(livenessInput.value) * 100
                )}%`;
        }

        if (
            arcfaceInput &&
            arcfaceValue
        ) {
            arcfaceValue.textContent =
                `${Math.round(
                    Number(arcfaceInput.value) * 100
                )}%`;
        }
    }


    recognitionInput?.addEventListener(
        "input",
        updateThresholdLabels
    );

    livenessInput?.addEventListener(
        "input",
        updateThresholdLabels
    );

    arcfaceInput?.addEventListener(
        "input",
        updateThresholdLabels
    );


    // Synchronize badges immediately on page load.
    updateThresholdLabels();


    form.addEventListener(
        "submit",
        async (event) => {
            event.preventDefault();

            const recognitionThreshold =
                Number(recognitionInput?.value);

            const livenessThreshold =
                Number(livenessInput?.value);

            const arcfaceThreshold =
                Number(arcfaceInput?.value);

            const cameraIndex =
                Number(cameraInput?.value);


            if (
                !Number.isFinite(recognitionThreshold) ||
                recognitionThreshold < 0 ||
                recognitionThreshold > 1
            ) {
                showToast(
                    "Recognition threshold is invalid.",
                    "error"
                );
                return;
            }


            if (
                !Number.isFinite(livenessThreshold) ||
                livenessThreshold < 0 ||
                livenessThreshold > 1
            ) {
                showToast(
                    "Liveness threshold is invalid.",
                    "error"
                );
                return;
            }


            if (
                !Number.isFinite(arcfaceThreshold) ||
                arcfaceThreshold < 0 ||
                arcfaceThreshold > 1
            ) {
                showToast(
                    "ArcFace verification threshold is invalid.",
                    "error"
                );
                return;
            }


            if (
                !Number.isInteger(cameraIndex) ||
                cameraIndex < 0
            ) {
                showToast(
                    "Camera index must be a non-negative integer.",
                    "error"
                );
                return;
            }


            const payload = {
                recognition_confidence_threshold:
                    recognitionThreshold,

                liveness_score_threshold:
                    livenessThreshold,

                arcface_verification_threshold:
                    arcfaceThreshold,

                attendance_once_per_day:
                    Boolean(attendanceInput?.checked),

                camera_index:
                    cameraIndex
            };


            const originalButtonHTML =
                saveButton?.innerHTML;

            if (saveButton) {
                saveButton.disabled = true;
                saveButton.textContent = "Saving...";
            }


            try {
                const response = await fetch(
                    "/api/settings",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",

                            "Accept":
                                "application/json"
                        },

                        body: JSON.stringify(payload)
                    }
                );

                const data =
                    await readJsonResponse(response);


                if (!response.ok) {
                    throw new Error(
                        data.detail ||
                        data.message ||
                        "Unable to save settings."
                    );
                }


                // Use backend-confirmed saved values.
                if (data.settings) {
                    if (
                        recognitionInput &&
                        data.settings
                            .recognition_confidence_threshold
                            !== undefined
                    ) {
                        recognitionInput.value =
                            data.settings
                                .recognition_confidence_threshold;
                    }

                    if (
                        livenessInput &&
                        data.settings
                            .liveness_score_threshold
                            !== undefined
                    ) {
                        livenessInput.value =
                            data.settings
                                .liveness_score_threshold;
                    }

                    if (
                        arcfaceInput &&
                        data.settings
                            .arcface_verification_threshold
                            !== undefined
                    ) {
                        arcfaceInput.value =
                            data.settings
                                .arcface_verification_threshold;
                    }

                    if (
                        attendanceInput &&
                        data.settings
                            .attendance_once_per_day
                            !== undefined
                    ) {
                        attendanceInput.checked =
                            Boolean(
                                data.settings
                                    .attendance_once_per_day
                            );
                    }

                    if (
                        cameraInput &&
                        data.settings.camera_index
                            !== undefined
                    ) {
                        cameraInput.value =
                            data.settings.camera_index;
                    }
                }


                updateThresholdLabels();

                showToast(
                    data.message ||
                    "Settings saved successfully.",
                    "success"
                );

            } catch (error) {
                showToast(
                    error.message ||
                    "Unable to save settings.",
                    "error"
                );

            } finally {
                if (saveButton) {
                    saveButton.disabled = false;

                    if (originalButtonHTML !== undefined) {
                        saveButton.innerHTML =
                            originalButtonHTML;
                    }
                }
            }
        }
    );
}