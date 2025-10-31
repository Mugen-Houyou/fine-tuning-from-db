/**
 * Main Application Logic
 * Korean Predictor Web Frontend
 */

// Global state
let api = null;
let predictionHistory = [];

// DOM Elements
const elements = {
    // Settings
    apiUrl: document.getElementById('api-url'),
    apiKey: document.getElementById('api-key'),
    modelSelect: document.getElementById('model-select'),
    topK: document.getElementById('top-k'),
    topKValue: document.getElementById('top-k-value'),
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temperature-value'),
    timeout: document.getElementById('timeout'),
    timeoutValue: document.getElementById('timeout-value'),
    completeWord: document.getElementById('complete-word'),
    includeSpecial: document.getElementById('include-special'),
    tempWarning: document.getElementById('temp-warning'),
    checkHealthBtn: document.getElementById('check-health'),

    // Input
    inputText: document.getElementById('input-text'),
    predictBtn: document.getElementById('predict-btn'),
    clearBtn: document.getElementById('clear-btn'),

    // Results
    loading: document.getElementById('loading'),
    errorMessage: document.getElementById('error-message'),
    results: document.getElementById('results'),

    // History
    history: document.getElementById('history'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),

    // Toast
    toast: document.getElementById('toast')
};

/**
 * Initialize application
 */
function init() {
    // Load saved settings
    loadSettings();

    // Initialize API client
    api = new KoreanPredictorAPI(elements.apiUrl.value, elements.apiKey.value);

    // Load history
    loadHistory();

    // Setup event listeners
    setupEventListeners();

    // Update slider displays
    updateSliderDisplays();

    // Check model-specific settings
    updateModelSpecificSettings();

    console.log('Korean Predictor 초기화 완료');
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Settings changes
    elements.apiUrl.addEventListener('change', () => {
        api.setConfig(elements.apiUrl.value, elements.apiKey.value);
        saveSettings();
    });

    elements.apiKey.addEventListener('change', () => {
        api.setConfig(elements.apiUrl.value, elements.apiKey.value);
        saveSettings();
    });

    elements.modelSelect.addEventListener('change', () => {
        updateModelSpecificSettings();
        saveSettings();
    });

    // Sliders
    elements.topK.addEventListener('input', () => {
        elements.topKValue.textContent = elements.topK.value;
        saveSettings();
    });

    elements.temperature.addEventListener('input', () => {
        elements.temperatureValue.textContent = elements.temperature.value;
        saveSettings();
    });

    elements.timeout.addEventListener('input', () => {
        elements.timeoutValue.textContent = elements.timeout.value;
        saveSettings();
    });

    // Checkboxes
    elements.completeWord.addEventListener('change', saveSettings);
    elements.includeSpecial.addEventListener('change', saveSettings);

    // Buttons
    elements.checkHealthBtn.addEventListener('click', checkHealth);
    elements.predictBtn.addEventListener('click', predict);
    elements.clearBtn.addEventListener('click', clearInput);
    elements.clearHistoryBtn.addEventListener('click', clearHistory);

    // Input
    elements.inputText.addEventListener('keydown', (e) => {
        // Ctrl+Enter to predict
        if (e.ctrlKey && e.key === 'Enter') {
            predict();
        }
    });
}

/**
 * Update model-specific UI elements
 */
function updateModelSpecificSettings() {
    const model = elements.modelSelect.value;
    const isReasoningModel = model === 'dna-r1';

    if (isReasoningModel) {
        // DNA-R1 doesn't support temperature
        elements.temperature.disabled = true;
        elements.tempWarning.classList.remove('hidden');
    } else {
        elements.temperature.disabled = false;
        elements.tempWarning.classList.add('hidden');
    }
}

/**
 * Update slider display values
 */
function updateSliderDisplays() {
    elements.topKValue.textContent = elements.topK.value;
    elements.temperatureValue.textContent = elements.temperature.value;
    elements.timeoutValue.textContent = elements.timeout.value;
}

/**
 * Check server health
 */
async function checkHealth() {
    const btn = elements.checkHealthBtn;
    btn.disabled = true;
    btn.textContent = '확인 중...';

    try {
        const status = await api.getStatus();
        showToast(`서버 정상 동작 중\n모델: ${status.model}\n가동 시간: ${formatUptime(status.uptime)}`, 'success');
    } catch (error) {
        showToast(`서버 연결 실패: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🏥 서버 상태 확인';
    }
}

/**
 * Format uptime in seconds to readable string
 */
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
}

/**
 * Main prediction function
 */
async function predict() {
    const text = elements.inputText.value.trim();

    if (!text) {
        showToast('텍스트를 입력하세요', 'warning');
        return;
    }

    // Show loading
    showLoading();
    hideError();
    elements.predictBtn.disabled = true;

    try {
        const options = {
            model: elements.modelSelect.value,
            top_k: parseInt(elements.topK.value),
            temperature: parseFloat(elements.temperature.value),
            complete_word: elements.completeWord.checked,
            include_special_tokens: elements.includeSpecial.checked,
            timeout: parseInt(elements.timeout.value)
        };

        const result = await api.predict(text, options);

        // Display results
        displayResults(result);

        // Add to history
        addToHistory(text, result);

        showToast('예측 완료!', 'success');
    } catch (error) {
        showError(error.message);
        showToast(`예측 실패: ${error.message}`, 'error');
    } finally {
        hideLoading();
        elements.predictBtn.disabled = false;
    }
}

/**
 * Display prediction results
 */
function displayResults(result) {
    const container = elements.results;
    container.innerHTML = '';

    // Create result card
    const card = document.createElement('div');
    card.className = 'result-card';

    // Header
    const header = document.createElement('div');
    header.className = 'result-header';

    const meta = document.createElement('div');
    meta.className = 'result-meta';
    meta.innerHTML = `
        <strong>입력:</strong> "${result.text}" <br>
        <strong>모델:</strong> ${result.model} |
        <strong>예측 시간:</strong> ${result.elapsed_time.toFixed(3)}초 |
        <strong>캐시:</strong> ${result.cached ? 'HIT ✓' : 'MISS'}
    `;

    const eotProb = document.createElement('div');
    eotProb.className = 'eot-probability';
    const eotPercent = (result.eot_probability * 100).toFixed(2);

    if (result.eot_probability > 0.7) {
        eotProb.classList.add('eot-high');
    } else if (result.eot_probability > 0.3) {
        eotProb.classList.add('eot-medium');
    } else {
        eotProb.classList.add('eot-low');
    }

    eotProb.textContent = `EOT: ${eotPercent}%`;

    header.appendChild(meta);
    header.appendChild(eotProb);

    // Predictions grid
    const grid = document.createElement('div');
    grid.className = 'predictions-grid';

    result.predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        // Rank
        const rank = document.createElement('div');
        rank.className = 'prediction-rank';
        rank.textContent = `#${index + 1}`;

        // Token
        const token = document.createElement('div');
        token.className = 'prediction-token';

        // Check if special token
        if (pred.token.startsWith('<') && pred.token.endsWith('>')) {
            token.classList.add('special');
        }
        if (pred.token === '</s>' || pred.token === '<|endoftext|>') {
            token.classList.add('eos');
        }

        token.textContent = pred.token;

        // Probability
        const prob = document.createElement('div');
        prob.className = 'prediction-probability';
        prob.textContent = `${(pred.probability * 100).toFixed(2)}%`;

        // Visual bar
        const bar = document.createElement('div');
        bar.className = 'prediction-bar';
        const barFill = document.createElement('div');
        barFill.className = 'prediction-bar-fill';
        barFill.style.width = `${pred.probability * 100}%`;
        bar.appendChild(barFill);

        item.appendChild(rank);
        item.appendChild(token);
        item.appendChild(prob);
        item.appendChild(bar);

        grid.appendChild(item);
    });

    card.appendChild(header);
    card.appendChild(grid);
    container.appendChild(card);
}

/**
 * Show loading state
 */
function showLoading() {
    elements.loading.classList.remove('hidden');
    elements.results.innerHTML = '';
}

/**
 * Hide loading state
 */
function hideLoading() {
    elements.loading.classList.add('hidden');
}

/**
 * Show error message
 */
function showError(message) {
    elements.errorMessage.textContent = `❌ ${message}`;
    elements.errorMessage.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    elements.errorMessage.classList.add('hidden');
}

/**
 * Clear input field
 */
function clearInput() {
    elements.inputText.value = '';
    elements.results.innerHTML = '';
    hideError();
}

/**
 * Add prediction to history
 */
function addToHistory(text, result) {
    const historyItem = {
        text: text,
        model: result.model,
        eot_probability: result.eot_probability,
        timestamp: new Date().toISOString(),
        result: result
    };

    predictionHistory.unshift(historyItem);

    // Keep only last 20 items
    if (predictionHistory.length > 20) {
        predictionHistory = predictionHistory.slice(0, 20);
    }

    saveHistory();
    renderHistory();
}

/**
 * Render history items
 */
function renderHistory() {
    const container = elements.history;
    container.innerHTML = '';

    if (predictionHistory.length === 0) {
        container.innerHTML = '<p style="color: #7f8c8d; text-align: center; padding: 20px;">히스토리가 비어있습니다</p>';
        return;
    }

    predictionHistory.forEach((item, index) => {
        const historyDiv = document.createElement('div');
        historyDiv.className = 'history-item';

        const textDiv = document.createElement('div');
        textDiv.className = 'history-text';
        textDiv.textContent = item.text.length > 50 ? item.text.substring(0, 50) + '...' : item.text;

        const metaDiv = document.createElement('div');
        metaDiv.className = 'history-meta';
        const time = new Date(item.timestamp).toLocaleString('ko-KR');
        metaDiv.textContent = `${time} | ${item.model} | EOT: ${(item.eot_probability * 100).toFixed(1)}%`;

        historyDiv.appendChild(textDiv);
        historyDiv.appendChild(metaDiv);

        // Click to reload
        historyDiv.addEventListener('click', () => {
            elements.inputText.value = item.text;
            displayResults(item.result);
            showToast('히스토리에서 복원됨', 'success');
        });

        container.appendChild(historyDiv);
    });
}

/**
 * Clear history
 */
function clearHistory() {
    if (predictionHistory.length === 0) {
        return;
    }

    if (confirm('히스토리를 모두 삭제하시겠습니까?')) {
        predictionHistory = [];
        saveHistory();
        renderHistory();
        showToast('히스토리가 삭제되었습니다', 'success');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toast = elements.toast;
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3000);
}

/**
 * Save settings to localStorage
 */
function saveSettings() {
    const settings = {
        apiUrl: elements.apiUrl.value,
        apiKey: elements.apiKey.value,
        model: elements.modelSelect.value,
        topK: elements.topK.value,
        temperature: elements.temperature.value,
        timeout: elements.timeout.value,
        completeWord: elements.completeWord.checked,
        includeSpecial: elements.includeSpecial.checked
    };

    localStorage.setItem('korean-predictor-settings', JSON.stringify(settings));
}

/**
 * Load settings from localStorage
 */
function loadSettings() {
    const saved = localStorage.getItem('korean-predictor-settings');
    if (!saved) return;

    try {
        const settings = JSON.parse(saved);

        if (settings.apiUrl) elements.apiUrl.value = settings.apiUrl;
        if (settings.apiKey) elements.apiKey.value = settings.apiKey;
        if (settings.model) elements.modelSelect.value = settings.model;
        if (settings.topK) elements.topK.value = settings.topK;
        if (settings.temperature) elements.temperature.value = settings.temperature;
        if (settings.timeout) elements.timeout.value = settings.timeout;
        if (settings.completeWord !== undefined) elements.completeWord.checked = settings.completeWord;
        if (settings.includeSpecial !== undefined) elements.includeSpecial.checked = settings.includeSpecial;
    } catch (e) {
        console.error('설정 로드 실패:', e);
    }
}

/**
 * Save history to localStorage
 */
function saveHistory() {
    localStorage.setItem('korean-predictor-history', JSON.stringify(predictionHistory));
}

/**
 * Load history from localStorage
 */
function loadHistory() {
    const saved = localStorage.getItem('korean-predictor-history');
    if (!saved) return;

    try {
        predictionHistory = JSON.parse(saved);
        renderHistory();
    } catch (e) {
        console.error('히스토리 로드 실패:', e);
        predictionHistory = [];
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
