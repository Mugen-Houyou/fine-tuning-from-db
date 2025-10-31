/**
 * API Communication Layer
 * Korean Predictor REST API Client
 */

class KoreanPredictorAPI {
    constructor(baseURL = 'http://localhost:8000', apiKey = '') {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
    }

    /**
     * Update API configuration
     */
    setConfig(baseURL, apiKey) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
    }

    /**
     * Make authenticated API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        // Add API key if provided
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        const config = {
            ...options,
            headers
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw {
                    status: response.status,
                    message: data.detail?.message || data.detail || 'API 요청 실패',
                    data: data
                };
            }

            return data;
        } catch (error) {
            if (error.status) {
                throw error;
            }
            // Network error
            throw {
                status: 0,
                message: '서버에 연결할 수 없습니다. API URL과 서버 상태를 확인하세요.',
                data: null
            };
        }
    }

    /**
     * Health check
     */
    async checkHealth() {
        return await this.request('/health', { method: 'GET' });
    }

    /**
     * Get detailed status
     */
    async getStatus() {
        return await this.request('/health/status', { method: 'GET' });
    }

    /**
     * Single text prediction
     */
    async predict(text, options = {}) {
        const payload = {
            text: text,
            model: options.model || 'kogpt2',
            top_k: options.top_k !== undefined ? options.top_k : 10,
            temperature: options.temperature !== undefined ? options.temperature : 1.3,
            complete_word: options.complete_word !== undefined ? options.complete_word : true,
            include_special_tokens: options.include_special_tokens !== undefined ? options.include_special_tokens : true,
            timeout: options.timeout !== undefined ? options.timeout : 60
        };

        return await this.request('/predict', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * Context-based prediction
     */
    async predictWithContext(context, currentText, options = {}) {
        const payload = {
            context: context,
            current_text: currentText,
            model: options.model || 'kogpt2',
            top_k: options.top_k !== undefined ? options.top_k : 10,
            temperature: options.temperature !== undefined ? options.temperature : 1.3,
            complete_word: options.complete_word !== undefined ? options.complete_word : true,
            include_special_tokens: options.include_special_tokens !== undefined ? options.include_special_tokens : true,
            timeout: options.timeout !== undefined ? options.timeout : 60
        };

        return await this.request('/predict/context', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * Batch prediction
     */
    async predictBatch(texts, options = {}) {
        const payload = {
            texts: texts,
            model: options.model || 'kogpt2',
            top_k: options.top_k !== undefined ? options.top_k : 10,
            temperature: options.temperature !== undefined ? options.temperature : 1.3,
            complete_word: options.complete_word !== undefined ? options.complete_word : true,
            include_special_tokens: options.include_special_tokens !== undefined ? options.include_special_tokens : true,
            timeout: options.timeout !== undefined ? options.timeout : 60
        };

        return await this.request('/predict/batch', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * List available models
     */
    async listModels() {
        return await this.request('/models', { method: 'GET' });
    }

    /**
     * Load a specific model
     */
    async loadModel(modelName) {
        return await this.request(`/models/${modelName}/load`, {
            method: 'POST'
        });
    }

    /**
     * Unload current model
     */
    async unloadModel() {
        return await this.request('/models/unload', {
            method: 'POST'
        });
    }

    /**
     * Get cache statistics
     */
    async getCacheStats() {
        return await this.request('/cache/stats', { method: 'GET' });
    }

    /**
     * Clear cache
     */
    async clearCache() {
        return await this.request('/cache/clear', {
            method: 'POST'
        });
    }
}

// Export for use in app.js
window.KoreanPredictorAPI = KoreanPredictorAPI;
