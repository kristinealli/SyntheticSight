(() => {
	"use strict";

	const config = window.SYNTHETIC_SIGHT_CONFIG || {};
	const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
	const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

	const fileInput = document.querySelector("#file-input");
	const dropZone = document.querySelector("#drop-zone");
	const dropPrompt = document.querySelector("#drop-prompt");
	const previewWrap = document.querySelector("#preview-wrap");
	const imagePreview = document.querySelector("#image-preview");
	const changeImageButton = document.querySelector("#change-image");
	const analyzeButton = document.querySelector("#analyze-button");
	const statusBox = document.querySelector("#status");

	const resultEmpty = document.querySelector("#result-empty");
	const resultContent = document.querySelector("#result-content");
	const resultLabel = document.querySelector("#result-label");
	const resultSummary = document.querySelector("#result-summary");
	const resultBadge = document.querySelector("#result-badge");
	const syntheticScore = document.querySelector("#synthetic-score");
	const realScore = document.querySelector("#real-score");
	const syntheticMeter = document.querySelector("#synthetic-meter");
	const realMeter = document.querySelector("#real-meter");
	const threshold = document.querySelector("#threshold");
	const caveatText = document.querySelector("#caveat-text");
	const streamlitLink = document.querySelector("#streamlit-link");

	let selectedFile = null;
	let previewUrl = null;

	if (config.streamlitUrl) {
		streamlitLink.href = config.streamlitUrl;
	}

	function setStatus(message = "", isError = false) {
		statusBox.textContent = message;
		statusBox.classList.toggle("is-error", isError);
	}

	function formatPercent(value) {
		const numeric = Number(value);
		if (!Number.isFinite(numeric)) return "—";
		return `${(numeric * 100).toFixed(2)}%`;
	}

	function validateFile(file) {
		if (!file) return "Choose an image first.";
		if (!ALLOWED_TYPES.has(file.type)) {
			return "Use a JPG, PNG, or WebP image.";
		}
		if (file.size > MAX_UPLOAD_BYTES) {
			return "The image is larger than the 10 MB upload limit.";
		}
		return null;
	}

	function clearPreviewUrl() {
		if (previewUrl) {
			URL.revokeObjectURL(previewUrl);
			previewUrl = null;
		}
	}

	function resetResult() {
		resultContent.hidden = true;
		resultEmpty.hidden = false;
		syntheticMeter.style.width = "0";
		realMeter.style.width = "0";
	}

	function setFile(file) {
		const error = validateFile(file);
		if (error) {
			selectedFile = null;
			analyzeButton.disabled = true;
			setStatus(error, true);
			return;
		}

		selectedFile = file;
		clearPreviewUrl();
		previewUrl = URL.createObjectURL(file);
		imagePreview.src = previewUrl;
		imagePreview.alt = `Preview of ${file.name}`;
		dropPrompt.hidden = true;
		previewWrap.hidden = false;
		analyzeButton.disabled = false;
		setStatus(
			`${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`,
		);
		resetResult();
	}

	function chooseFile() {
		fileInput.value = "";
		fileInput.click();
	}

	dropZone.addEventListener("click", (event) => {
		if (event.target === changeImageButton) return;
		chooseFile();
	});

	dropZone.addEventListener("keydown", (event) => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			chooseFile();
		}
	});

	changeImageButton.addEventListener("click", (event) => {
		event.stopPropagation();
		chooseFile();
	});

	fileInput.addEventListener("change", () => {
		setFile(fileInput.files?.[0]);
	});

	["dragenter", "dragover"].forEach((eventName) => {
		dropZone.addEventListener(eventName, (event) => {
			event.preventDefault();
			dropZone.classList.add("is-dragging");
		});
	});

	["dragleave", "drop"].forEach((eventName) => {
		dropZone.addEventListener(eventName, (event) => {
			event.preventDefault();
			dropZone.classList.remove("is-dragging");
		});
	});

	dropZone.addEventListener("drop", (event) => {
		setFile(event.dataTransfer?.files?.[0]);
	});

	function renderResult(data) {
		const isSynthetic = String(data.label).toLowerCase() === "synthetic";
		const synthetic = Number(data.synthetic_probability);
		const real = Number(data.real_probability);
		const decisionThreshold = Number(data.decision_threshold);

		resultLabel.textContent = isSynthetic ? "Synthetic" : "Real";
		resultSummary.textContent = isSynthetic
			? "The model's synthetic score met or exceeded the operating threshold."
			: "The model's synthetic score remained below the operating threshold.";

		resultBadge.textContent = "Review signal";
		resultBadge.className = `result-badge ${isSynthetic ? "synthetic" : "real"}`;

		syntheticScore.textContent = formatPercent(synthetic);
		realScore.textContent = formatPercent(real);
		syntheticMeter.style.width = `${Math.max(0, Math.min(100, synthetic * 100))}%`;
		realMeter.style.width = `${Math.max(0, Math.min(100, real * 100))}%`;
		threshold.textContent = Number.isFinite(decisionThreshold)
			? decisionThreshold.toFixed(2)
			: "—";
		caveatText.textContent =
			data.caveat ||
			"Review signal only; this prediction does not authenticate image provenance.";

		resultEmpty.hidden = true;
		resultContent.hidden = false;
	}

	analyzeButton.addEventListener("click", async () => {
		const error = validateFile(selectedFile);
		if (error) {
			setStatus(error, true);
			return;
		}

		const apiBaseUrl = String(config.apiBaseUrl || "").replace(/\/+$/, "");
		if (!apiBaseUrl) {
			setStatus(
				"The GitHub Pages frontend is ready, but an inference API URL has not been configured yet. Add it in frontend/config.js or use the Streamlit demo.",
				true,
			);
			return;
		}

		const form = new FormData();
		form.append("file", selectedFile);

		analyzeButton.disabled = true;
		analyzeButton.textContent = "Analyzing…";
		setStatus("Sending image to the Synthetic Sight inference API…");

		try {
			const response = await fetch(`${apiBaseUrl}/predict`, {
				method: "POST",
				body: form,
				credentials: "omit",
			});

			let payload = null;
			try {
				payload = await response.json();
			} catch {
				payload = null;
			}

			if (!response.ok) {
				const detail =
					payload?.detail ||
					`Request failed with status ${response.status}.`;
				throw new Error(detail);
			}

			renderResult(payload);
			setStatus("✓ Analysis complete.");
		} catch (error) {
			const message =
				error instanceof TypeError
					? "Could not reach the inference API. Check the API URL, HTTPS availability, and CORS configuration."
					: error.message;
			setStatus(message, true);
		} finally {
			analyzeButton.disabled = false;
			analyzeButton.textContent = resultContent.hidden
				? "Analyze image"
				: "Analyze again";
		}
	});

	window.addEventListener("beforeunload", clearPreviewUrl);
})();
