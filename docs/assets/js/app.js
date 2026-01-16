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

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadResults();
    populateModelSelects();
    setupEventListeners();
    await loadModelMetadata();
    updateComparison();
    setupMagnifiers();
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

    // Re-setup magnifiers
    setTimeout(setupMagnifiers, 100);
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

    // Heatmap toggle
    document.getElementById('heatmap-toggle').addEventListener('change', (e) => {
        const visible = e.target.checked;
        document.querySelectorAll('.heatmap-overlay').forEach(el => {
            el.classList.toggle('visible', visible);
        });
    });

    // Heatmap opacity
    document.getElementById('heatmap-opacity').addEventListener('input', (e) => {
        const opacity = e.target.value / 100;
        document.querySelectorAll('.heatmap-overlay').forEach(el => {
            el.style.opacity = opacity;
        });
    });
}

// Setup magnifier effect
function setupMagnifiers() {
    const panels = ['panel-original', 'panel-model-a', 'panel-model-b'];
    const imgIds = ['img-original', 'img-model-a', 'img-model-b'];
    const magIds = ['mag-original', 'mag-model-a', 'mag-model-b'];

    panels.forEach((panelId, idx) => {
        const container = document.querySelector(`#${panelId} .image-container`);
        const img = document.getElementById(imgIds[idx]);
        const mag = document.getElementById(magIds[idx]);

        if (!container || !img || !mag) return;

        // Remove old listeners by cloning
        const newContainer = container.cloneNode(true);
        container.parentNode.replaceChild(newContainer, container);

        const newImg = document.getElementById(imgIds[idx]);
        const newMag = document.getElementById(magIds[idx]);

        newContainer.addEventListener('mousemove', (e) => {
            const rect = newContainer.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const magSize = 150;
            newMag.style.left = (x - magSize/2) + 'px';
            newMag.style.top = (y - magSize/2) + 'px';

            const zoom = 2.5;
            const bgX = -(x * zoom - magSize/2);
            const bgY = -(y * zoom - magSize/2);

            newMag.style.backgroundImage = `url(${newImg.src})`;
            newMag.style.backgroundSize = `${rect.width * zoom}px ${rect.height * zoom}px`;
            newMag.style.backgroundPosition = `${bgX}px ${bgY}px`;
        });

        newContainer.addEventListener('mouseenter', () => {
            newMag.style.opacity = '1';
        });

        newContainer.addEventListener('mouseleave', () => {
            newMag.style.opacity = '0';
        });
    });
}
