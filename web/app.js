/**
 * vLLM VRAM Calculator - Frontend Application
 * 
 * Handles UI interactions and API communication for the VRAM calculator.
 */

// API base URL - adjust for production deployment
const API_BASE = '';

// State
let modelsData = [];
let gpusData = [];
let currentResult = null;
let debounceTimer = null;

// DOM Elements
const elements = {
    modelSelect: document.getElementById('model-select'),
    quantizationSelect: document.getElementById('quantization-select'),
    kvCacheSelect: document.getElementById('kv-cache-select'),
    gpuSelect: document.getElementById('gpu-select'),
    numGpus: document.getElementById('num-gpus'),
    customVramGroup: document.getElementById('custom-vram-group'),
    customVram: document.getElementById('custom-vram'),
    batchSize: document.getElementById('batch-size'),
    batchSizeValue: document.getElementById('batch-size-value'),
    sequenceLength: document.getElementById('sequence-length'),
    sequenceLengthValue: document.getElementById('sequence-length-value'),
    concurrentUsers: document.getElementById('concurrent-users'),
    concurrentUsersValue: document.getElementById('concurrent-users-value'),
    sliderToggle: document.getElementById('slider-toggle'),
    gaugeFill: document.getElementById('gauge-fill'),
    gaugePercent: document.getElementById('gauge-percent'),
    statusBadge: document.getElementById('status-badge'),
    totalVram: document.getElementById('total-vram'),
    vramOfTotal: document.getElementById('vram-of-total'),
    sharedMemory: document.getElementById('shared-memory'),
    perUserMemory: document.getElementById('per-user-memory'),
    concurrentHint: document.getElementById('concurrent-hint'),
    modelInfoName: document.getElementById('model-info-name'),
    infoWeights: document.getElementById('info-weights'),
    infoKvCache: document.getElementById('info-kv-cache'),
    infoAttention: document.getElementById('info-attention'),
    modeInfo: document.getElementById('mode-info'),
    maxUsers: document.getElementById('max-users'),
    // Allocation elements
    allocWeights: document.getElementById('alloc-weights'),
    allocKvCache: document.getElementById('alloc-kv-cache'),
    allocActivations: document.getElementById('alloc-activations'),
    allocOverhead: document.getElementById('alloc-overhead'),
    legendWeights: document.getElementById('legend-weights'),
    legendWeightsPct: document.getElementById('legend-weights-pct'),
    legendKvCache: document.getElementById('legend-kv-cache'),
    legendKvCachePct: document.getElementById('legend-kv-cache-pct'),
    legendActivations: document.getElementById('legend-activations'),
    legendActivationsPct: document.getElementById('legend-activations-pct'),
    legendOverhead: document.getElementById('legend-overhead'),
    legendOverheadPct: document.getElementById('legend-overhead-pct'),
};

// Utility Functions
function formatNumber(num) {
    if (num >= 1000) {
        return num.toLocaleString();
    }
    return num.toString();
}

function formatGB(gb) {
    if (gb >= 1000) {
        return `${(gb / 1000).toFixed(2)} TB`;
    }
    return `${gb.toFixed(2)} GB`;
}

function formatSequenceLength(value) {
    if (value >= 1000) {
        return `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}K`;
    }
    return value.toString();
}

function debounce(func, wait) {
    return function executedFunction(...args) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => func.apply(this, args), wait);
    };
}

// API Functions
async function fetchModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        if (!response.ok) throw new Error('Failed to fetch models');
        const data = await response.json();
        modelsData = data.models;
        populateModelSelect();
    } catch (error) {
        console.error('Error fetching models:', error);
        elements.modelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
}

async function fetchGPUs() {
    try {
        const response = await fetch(`${API_BASE}/api/gpus`);
        if (!response.ok) throw new Error('Failed to fetch GPUs');
        const data = await response.json();
        gpusData = data.gpus;
        populateGPUSelect();
    } catch (error) {
        console.error('Error fetching GPUs:', error);
        elements.gpuSelect.innerHTML = '<option value="">Error loading GPUs</option>';
    }
}

async function calculateVRAM() {
    const modelId = elements.modelSelect.value;
    const gpuId = elements.gpuSelect.value;
    
    if (!modelId || !gpuId) {
        return;
    }
    
    setCalculating(true);
    
    const request = {
        model_id: modelId,
        gpu_id: gpuId,
        quantization: elements.quantizationSelect.value,
        kv_cache_quantization: elements.kvCacheSelect.value,
        batch_size: parseInt(elements.batchSize.value),
        sequence_length: parseInt(elements.sequenceLength.value),
        concurrent_users: parseInt(elements.concurrentUsers.value),
        num_gpus: parseInt(elements.numGpus.value),
    };
    
    if (gpuId === 'custom') {
        request.custom_vram_gb = parseFloat(elements.customVram.value);
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/calculate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Calculation failed');
        }
        
        currentResult = await response.json();
        updateResults(currentResult);
    } catch (error) {
        console.error('Error calculating VRAM:', error);
        setError(error.message);
    } finally {
        setCalculating(false);
    }
}

// UI Functions
function populateModelSelect() {
    elements.modelSelect.innerHTML = '<option value="">Select a model...</option>';
    
    // Group models by provider
    const providers = {};
    modelsData.forEach(model => {
        if (!providers[model.provider]) {
            providers[model.provider] = [];
        }
        providers[model.provider].push(model);
    });
    
    // Create optgroups
    Object.keys(providers).sort().forEach(provider => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = provider;
        
        providers[provider].forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            const params = model.parameters >= 1e9 
                ? `${(model.parameters / 1e9).toFixed(1)}B`
                : `${(model.parameters / 1e6).toFixed(0)}M`;
            option.textContent = `${model.name} (${params})`;
            optgroup.appendChild(option);
        });
        
        elements.modelSelect.appendChild(optgroup);
    });
}

function populateGPUSelect() {
    elements.gpuSelect.innerHTML = '';
    
    // Group GPUs by vendor
    const vendors = {};
    gpusData.forEach(gpu => {
        if (!vendors[gpu.vendor]) {
            vendors[gpu.vendor] = [];
        }
        vendors[gpu.vendor].push(gpu);
    });
    
    // Create optgroups
    Object.keys(vendors).sort().forEach(vendor => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = vendor;
        
        vendors[vendor].forEach(gpu => {
            const option = document.createElement('option');
            option.value = gpu.id;
            option.textContent = `${gpu.name} (${gpu.vram_gb}GB)`;
            optgroup.appendChild(option);
        });
        
        elements.gpuSelect.appendChild(optgroup);
    });
    
    // Select a sensible default (RTX 4090 if available)
    const defaultGpu = gpusData.find(g => g.id === 'rtx-4090-24gb');
    if (defaultGpu) {
        elements.gpuSelect.value = defaultGpu.id;
    } else if (gpusData.length > 0) {
        elements.gpuSelect.value = gpusData[0].id;
    }
}

function updateSliderValue(slider, valueElement, formatFn = formatNumber) {
    const value = parseInt(slider.value);
    valueElement.textContent = formatFn(value);
}

function setCalculating(isCalculating) {
    if (isCalculating) {
        elements.statusBadge.textContent = 'CALCULATING...';
        elements.statusBadge.className = 'status-badge calculating';
    }
}

function setError(message) {
    elements.statusBadge.textContent = 'ERROR';
    elements.statusBadge.className = 'status-badge insufficient';
    console.error(message);
}

function updateResults(result) {
    // Update gauge
    const percent = Math.min(result.vram_utilization_percent, 100);
    const circumference = 2 * Math.PI * 85;
    const offset = circumference - (percent / 100) * circumference;
    
    elements.gaugeFill.style.strokeDashoffset = offset;
    elements.gaugePercent.textContent = `${percent.toFixed(1)}%`;
    
    // Update gauge color based on utilization
    elements.gaugeFill.classList.remove('warning', 'error');
    if (percent > 100 || !result.fits_in_memory) {
        elements.gaugeFill.classList.add('error');
    } else if (percent > 85) {
        elements.gaugeFill.classList.add('warning');
    }
    
    // Update status badge
    if (!result.fits_in_memory) {
        elements.statusBadge.textContent = 'INSUFFICIENT';
        elements.statusBadge.className = 'status-badge insufficient';
    } else if (percent > 85) {
        elements.statusBadge.textContent = 'TIGHT FIT';
        elements.statusBadge.className = 'status-badge warning';
    } else {
        elements.statusBadge.textContent = 'SUFFICIENT';
        elements.statusBadge.className = 'status-badge sufficient';
    }
    
    // Update total VRAM display
    elements.totalVram.textContent = formatGB(result.memory.total_gb);
    elements.vramOfTotal.textContent = `of ${result.gpu_vram_gb} GB VRAM`;
    
    // Update per-user breakdown
    elements.sharedMemory.textContent = `${formatGB(result.shared_memory_gb)} shared`;
    elements.perUserMemory.textContent = `${formatGB(result.per_user_memory_gb)} per user`;
    
    const users = parseInt(elements.concurrentUsers.value);
    elements.concurrentHint.textContent = `Total for ${users} concurrent user${users > 1 ? 's' : ''}`;
    
    // Update model info
    elements.modelInfoName.textContent = result.model_name;
    elements.infoWeights.textContent = elements.quantizationSelect.value.toUpperCase();
    elements.infoKvCache.textContent = elements.kvCacheSelect.value.toUpperCase();
    
    // Get attention type from model data
    const model = modelsData.find(m => m.id === elements.modelSelect.value);
    if (model) {
        elements.infoAttention.textContent = model.attention_type.toUpperCase();
    }
    
    // Update mode info
    const batch = elements.batchSize.value;
    elements.modeInfo.textContent = `Mode: Inference | Batch: ${batch} | Users: ${users}`;
    
    // Update max users
    elements.maxUsers.textContent = result.max_concurrent_users;
    
    // Update allocation chart
    updateAllocationChart(result.memory);
}

function updateAllocationChart(memory) {
    const total = memory.total_gb;
    if (total === 0) return;
    
    const weightsPct = (memory.weights_gb / total) * 100;
    const kvCachePct = (memory.kv_cache_gb / total) * 100;
    const activationsPct = (memory.activations_gb / total) * 100;
    const overheadPct = (memory.overhead_gb / total) * 100;
    
    // Update bar segments
    elements.allocWeights.style.width = `${weightsPct}%`;
    elements.allocKvCache.style.width = `${kvCachePct}%`;
    elements.allocActivations.style.width = `${activationsPct}%`;
    elements.allocOverhead.style.width = `${overheadPct}%`;
    
    // Update legend values
    elements.legendWeights.textContent = formatGB(memory.weights_gb);
    elements.legendWeightsPct.textContent = `${weightsPct.toFixed(1)}%`;
    
    elements.legendKvCache.textContent = formatGB(memory.kv_cache_gb);
    elements.legendKvCachePct.textContent = `${kvCachePct.toFixed(1)}%`;
    
    elements.legendActivations.textContent = formatGB(memory.activations_gb);
    elements.legendActivationsPct.textContent = `${activationsPct.toFixed(1)}%`;
    
    elements.legendOverhead.textContent = formatGB(memory.overhead_gb);
    elements.legendOverheadPct.textContent = `${overheadPct.toFixed(1)}%`;
}

// Event Handlers
function setupEventListeners() {
    // Model selection
    elements.modelSelect.addEventListener('change', () => {
        calculateVRAM();
    });
    
    // GPU selection
    elements.gpuSelect.addEventListener('change', () => {
        // Show/hide custom VRAM input
        if (elements.gpuSelect.value === 'custom') {
            elements.customVramGroup.style.display = 'block';
        } else {
            elements.customVramGroup.style.display = 'none';
        }
        calculateVRAM();
    });
    
    // Custom VRAM
    elements.customVram.addEventListener('change', () => {
        calculateVRAM();
    });
    
    // Quantization
    elements.quantizationSelect.addEventListener('change', () => {
        calculateVRAM();
    });
    
    // KV Cache quantization
    elements.kvCacheSelect.addEventListener('change', () => {
        calculateVRAM();
    });
    
    // Number of GPUs buttons
    document.querySelectorAll('.num-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.target);
            const action = btn.dataset.action;
            const min = parseInt(target.min);
            const max = parseInt(target.max);
            let value = parseInt(target.value);
            
            if (action === 'increase' && value < max) {
                target.value = value + 1;
            } else if (action === 'decrease' && value > min) {
                target.value = value - 1;
            }
            
            calculateVRAM();
        });
    });
    
    // Num GPUs direct input
    elements.numGpus.addEventListener('change', () => {
        calculateVRAM();
    });
    
    // Batch size slider
    elements.batchSize.addEventListener('input', () => {
        updateSliderValue(elements.batchSize, elements.batchSizeValue);
    });
    elements.batchSize.addEventListener('change', debounce(() => {
        calculateVRAM();
    }, 300));
    
    // Sequence length slider
    elements.sequenceLength.addEventListener('input', () => {
        updateSliderValue(elements.sequenceLength, elements.sequenceLengthValue, formatSequenceLength);
    });
    elements.sequenceLength.addEventListener('change', debounce(() => {
        calculateVRAM();
    }, 300));
    
    // Concurrent users slider
    elements.concurrentUsers.addEventListener('input', () => {
        updateSliderValue(elements.concurrentUsers, elements.concurrentUsersValue);
    });
    elements.concurrentUsers.addEventListener('change', debounce(() => {
        calculateVRAM();
    }, 300));
    
    // Mode toggle buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

// Initialize
async function init() {
    setupEventListeners();
    
    // Initialize slider values
    updateSliderValue(elements.batchSize, elements.batchSizeValue);
    updateSliderValue(elements.sequenceLength, elements.sequenceLengthValue, formatSequenceLength);
    updateSliderValue(elements.concurrentUsers, elements.concurrentUsersValue);
    
    // Fetch data
    await Promise.all([fetchModels(), fetchGPUs()]);
    
    // Select first model if available
    if (modelsData.length > 0) {
        // Default to a popular model like Llama 3.1 8B
        const defaultModel = modelsData.find(m => m.id === 'llama-3.1-8b') || modelsData[0];
        elements.modelSelect.value = defaultModel.id;
        
        // Trigger initial calculation
        calculateVRAM();
    }
}

// Start the application
document.addEventListener('DOMContentLoaded', init);
