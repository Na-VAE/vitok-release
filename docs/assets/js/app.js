// ViTok Evaluation Results - Interactive Viewer

// Configuration
const IMAGES_BASE = 'images';
const DATA_BASE = 'data';

// Available models
const VITOK_MODELS = ['5B-f16x64', '5B-f16x32', '5B-f32x128', '5B-f32x64'];
const BASELINE_MODELS = ['flux', 'sd'];
const ALL_MODELS = [...VITOK_MODELS, ...BASELINE_MODELS];

// State
let resultsData = {};
let modelMetadata = {};
let currentResolution = 'challenge-768';
let currentImageIdx = 0;
let currentModelA = '5B-f16x64';
let currentModelB = 'flux';
let heatmapVisible = false;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadResults();
    populateModelSelects();
    setupEventListeners();
    await loadModelMetadata();
    updateComparison();
});

// Load benchmark results
async function loadResults() {
    try {
        const response = await fetch(`${DATA_BASE}/results.json`);
        if (response.ok) {
            resultsData = await response.json();
            renderResultsTable('div8k-1024');
        }
    } catch (e) {
        console.log('Results not loaded:', e);
        document.getElementById('results-table-container').innerHTML =
            '<p class="loading">Results data not available.</p>';
    }
}

// Load metadata for current model/resolution
async function loadModelMetadata() {
    modelMetadata = {};
    for (const model of ALL_MODELS) {
        try {
            const url = `${IMAGES_BASE}/${currentResolution}/${model}/metadata.json`;
            const response = await fetch(url);
            if (response.ok) {
                modelMetadata[model] = await response.json();
            }
        } catch (e) {
            console.log(`Metadata not found for ${model}`);
        }
    }
}

// Render results table
function renderResultsTable(config) {
    const container = document.getElementById('results-table-container');

    if (!resultsData[config] || resultsData[config].length === 0) {
        container.innerHTML = '<p class="loading">No results for this configuration.</p>';
        return;
    }

    const data = resultsData[config];
    const metrics = ['psnr', 'ssim', 'fid', 'fdd'];

    // Find best values
    const best = {};
    metrics.forEach(m => {
        const values = data.map(d => d[m]).filter(v => v !== null && v !== undefined);
        if (values.length === 0) return;
        if (m === 'fid' || m === 'fdd') {
            best[m] = Math.min(...values);
        } else {
            best[m] = Math.max(...values);
        }
    });

    let html = `
        <table class="results-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>PSNR ↑</th>
                    <th>SSIM ↑</th>
                    <th>FID ↓</th>
                    <th>FDD ↓</th>
                    <th>Throughput</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.forEach(row => {
        const isOOM = row.note === 'OOM';
        html += '<tr>';
        html += `<td>${row.model}</td>`;

        metrics.forEach(m => {
            const val = row[m];
            if (isOOM || val === null || val === undefined) {
                html += `<td class="oom">OOM</td>`;
            } else {
                const isBest = Math.abs(val - best[m]) < 0.001;
                const cls = isBest ? 'best' : '';
                html += `<td class="${cls}">${val.toFixed(2)}</td>`;
            }
        });

        const throughput = row.throughput_img_per_sec;
        if (isOOM || throughput === null || throughput === undefined) {
            html += `<td class="oom">-</td>`;
        } else {
            html += `<td>${throughput.toFixed(2)} img/s</td>`;
        }
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// Populate model select dropdowns
function populateModelSelects() {
    const selectA = document.getElementById('model-a-select');
    const selectB = document.getElementById('model-b-select');

    [selectA, selectB].forEach(select => {
        select.innerHTML = '';

        // ViTok models
        const vitokGroup = document.createElement('optgroup');
        vitokGroup.label = 'ViTok Models';
        VITOK_MODELS.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            vitokGroup.appendChild(opt);
        });
        select.appendChild(vitokGroup);

        // Baseline models
        const baselineGroup = document.createElement('optgroup');
        baselineGroup.label = 'Baseline VAEs';
        BASELINE_MODELS.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m.charAt(0).toUpperCase() + m.slice(1) + ' VAE';
            baselineGroup.appendChild(opt);
        });
        select.appendChild(baselineGroup);
    });

    selectA.value = currentModelA;
    selectB.value = currentModelB;
}

// Update visual comparison
function updateComparison() {
    const basePath = `${IMAGES_BASE}/${currentResolution}`;
    const imgNum = String(currentImageIdx).padStart(4, '0');

    // Original image (use model A's original)
    const imgOriginal = document.getElementById('img-original');
    imgOriginal.src = `${basePath}/${currentModelA}/originals/${imgNum}.jpg`;

    // Model A
    updateModelPanel('a', currentModelA, basePath, imgNum);

    // Model B
    updateModelPanel('b', currentModelB, basePath, imgNum);

    // Setup synchronized magnifiers after images update
    setTimeout(setupSyncedMagnifiers, 100);
}

function updateModelPanel(side, model, basePath, imgNum) {
    const img = document.getElementById(`img-model-${side}`);
    const heatmap = document.getElementById(`heatmap-model-${side}`);
    const metrics = document.getElementById(`metrics-model-${side}`);
    const label = document.getElementById(`label-model-${side}`);

    const displayName = BASELINE_MODELS.includes(model)
        ? model.charAt(0).toUpperCase() + model.slice(1) + ' VAE'
        : model;
    label.textContent = displayName;

    img.src = `${basePath}/${model}/recons/${imgNum}.jpg`;
    heatmap.src = `${basePath}/${model}/heatmaps_l1/${imgNum}.jpg`;

    // Apply current heatmap visibility
    heatmap.classList.toggle('visible', heatmapVisible);

    // Load metrics from metadata
    if (modelMetadata[model] && modelMetadata[model].images) {
        const imgData = modelMetadata[model].images[currentImageIdx];
        if (imgData) {
            metrics.innerHTML = `
                <span class="metric-label">PSNR:</span> <span class="metric-value">${imgData.psnr.toFixed(2)}</span>
                <span class="metric-label">SSIM:</span> <span class="metric-value">${imgData.ssim.toFixed(4)}</span>
                <span class="metric-label">L1:</span> <span class="metric-value">${imgData.l1.toFixed(4)}</span>
            `;
            return;
        }
    }
    metrics.innerHTML = '';
}

// Setup event listeners
function setupEventListeners() {
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderResultsTable(btn.dataset.tab);
        });
    });

    // Resolution select
    document.getElementById('resolution-select').addEventListener('change', async (e) => {
        currentResolution = e.target.value;
        await loadModelMetadata();
        updateComparison();
    });

    // Image select
    document.getElementById('image-select').addEventListener('change', (e) => {
        currentImageIdx = parseInt(e.target.value);
        updateComparison();
    });

    // Model selects
    document.getElementById('model-a-select').addEventListener('change', (e) => {
        currentModelA = e.target.value;
        updateComparison();
    });

    document.getElementById('model-b-select').addEventListener('change', (e) => {
        currentModelB = e.target.value;
        updateComparison();
    });

    // Heatmap toggle (simple on/off)
    document.getElementById('heatmap-toggle').addEventListener('change', (e) => {
        heatmapVisible = e.target.checked;
        document.querySelectorAll('.heatmap-overlay').forEach(el => {
            el.classList.toggle('visible', heatmapVisible);
        });
    });
}

// Setup synchronized magnifiers across all panels
function setupSyncedMagnifiers() {
    const containers = document.querySelectorAll('.image-container');
    const imgs = [
        document.getElementById('img-original'),
        document.getElementById('img-model-a'),
        document.getElementById('img-model-b')
    ];
    const mags = [
        document.getElementById('mag-original'),
        document.getElementById('mag-model-a'),
        document.getElementById('mag-model-b')
    ];

    const magSize = 360;
    const zoom = 3;

    containers.forEach((container, idx) => {
        // Clone to remove old listeners
        const newContainer = container.cloneNode(true);
        container.parentNode.replaceChild(newContainer, container);
    });

    // Re-get references after cloning
    const newContainers = document.querySelectorAll('.image-container');
    const newImgs = [
        document.getElementById('img-original'),
        document.getElementById('img-model-a'),
        document.getElementById('img-model-b')
    ];
    const newMags = [
        document.getElementById('mag-original'),
        document.getElementById('mag-model-a'),
        document.getElementById('mag-model-b')
    ];

    newContainers.forEach((container, sourceIdx) => {
        container.addEventListener('mousemove', (e) => {
            const rect = container.getBoundingClientRect();
            const xPct = (e.clientX - rect.left) / rect.width;
            const yPct = (e.clientY - rect.top) / rect.height;

            // Update ALL magnifiers at the same relative position
            newContainers.forEach((targetContainer, targetIdx) => {
                const targetRect = targetContainer.getBoundingClientRect();
                const targetImg = newImgs[targetIdx];
                const targetMag = newMags[targetIdx];

                if (!targetImg || !targetMag) return;

                // Position magnifier
                const x = xPct * targetRect.width;
                const y = yPct * targetRect.height;
                targetMag.style.left = (x - magSize / 2) + 'px';
                targetMag.style.top = (y - magSize / 2) + 'px';

                // Background position for zoom
                const bgX = -xPct * targetImg.naturalWidth * zoom + magSize / 2;
                const bgY = -yPct * targetImg.naturalHeight * zoom + magSize / 2;

                targetMag.style.backgroundImage = `url(${targetImg.src})`;
                targetMag.style.backgroundSize = `${targetImg.naturalWidth * zoom}px ${targetImg.naturalHeight * zoom}px`;
                targetMag.style.backgroundPosition = `${bgX}px ${bgY}px`;
                targetMag.style.opacity = '1';
            });
        });

        container.addEventListener('mouseleave', () => {
            // Hide ALL magnifiers
            newMags.forEach(mag => {
                if (mag) mag.style.opacity = '0';
            });
        });
    });
}
