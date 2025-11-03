/**
 * JavaScript for Image Colorization App
 * Handles file upload, API communication, and UI interactions
 */

class ImageColorizationApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.checkModelStatus();
    }

    /**
     * Initialize DOM elements
     */
    initializeElements() {
        this.uploadForm = document.getElementById('uploadForm');
        this.imageFile = document.getElementById('imageFile');
        this.colorizeBtn = document.getElementById('colorizeBtn');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.errorAlert = document.getElementById('errorAlert');
        this.errorMessage = document.getElementById('errorMessage');
        this.resultsSection = document.getElementById('resultsSection');
        
        // Result images
        this.originalImage = document.getElementById('originalImage');
        this.grayscaleImage = document.getElementById('grayscaleImage');
        this.colorizedImage = document.getElementById('colorizedImage');
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        this.uploadForm.addEventListener('submit', (e) => this.handleSubmit(e));
        this.imageFile.addEventListener('change', (e) => this.handleFileSelect(e));
    }

    /**
     * Check if the model is loaded and ready
     */
    async checkModelStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (!data.model_loaded) {
                this.showError('Model is not loaded yet. Please wait...');
                this.colorizeBtn.disabled = true;
            } else {
                console.log('Model is ready!');
                this.hideError();
                this.colorizeBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            this.showError('Unable to connect to the server. Please check your connection.');
        }
    }

    /**
     * Handle file selection
     */
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                this.showError('Please select a valid image file.');
                this.clearFileInput();
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                this.showError('File size must be less than 10MB.');
                this.clearFileInput();
                return;
            }

            this.hideError();
            this.hideResults();
            
            // Show preview if needed
            this.previewFile(file);
        }
    }

    /**
     * Preview selected file
     */
    previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Could add a preview section here if needed
            console.log('File selected:', file.name);
        };
        reader.readAsDataURL(file);
    }

    /**
     * Handle form submission
     */
    async handleSubmit(event) {
        event.preventDefault();
        
        const file = this.imageFile.files[0];
        if (!file) {
            this.showError('Please select an image file.');
            return;
        }

        try {
            this.showLoading();
            this.hideError();
            this.hideResults();

            // Create FormData
            const formData = new FormData();
            formData.append('file', file);

            // Make API request
            const response = await fetch('/colorize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error occurred');
            }

            const data = await response.json();
            
            if (data.success) {
                this.displayResults(data.images);
                this.showResults();
                this.scrollToResults();
            } else {
                throw new Error(data.message || 'Colorization failed');
            }

        } catch (error) {
            console.error('Error during colorization:', error);
            this.showError(error.message || 'Failed to colorize image. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Display colorization results
     */
    displayResults(images) {
        if (images.original) {
            this.originalImage.src = images.original;
            this.originalImage.alt = 'Original Image';
        }
        
        if (images.grayscale) {
            this.grayscaleImage.src = images.grayscale;
            this.grayscaleImage.alt = 'Grayscale Image';
        }
        
        if (images.colorized) {
            document.getElementById('colorizedImage').src = images.colorized;
            document.getElementById('colorizedImage').alt = 'AI Colorized Image';
        }

        // Add fade-in animation
        this.resultsSection.classList.add('fade-in-up');
    }

    /**
     * Show loading state
     */
    showLoading() {
        this.loadingSpinner.style.display = 'block';
        this.colorizeBtn.classList.add('btn-loading');
        this.colorizeBtn.disabled = true;
        this.imageFile.disabled = true;
    }

    /**
     * Hide loading state
     */
    hideLoading() {
        this.loadingSpinner.style.display = 'none';
        this.colorizeBtn.classList.remove('btn-loading');
        this.colorizeBtn.disabled = false;
        this.imageFile.disabled = false;
    }

    /**
     * Show error message
     */
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorAlert.style.display = 'block';
        this.scrollToError();
    }

    /**
     * Hide error message
     */
    hideError() {
        this.errorAlert.style.display = 'none';
    }

    /**
     * Show results section
     */
    showResults() {
        this.resultsSection.style.display = 'block';
    }

    /**
     * Hide results section
     */
    hideResults() {
        this.resultsSection.style.display = 'none';
    }

    /**
     * Clear file input
     */
    clearFileInput() {
        this.imageFile.value = '';
    }

    /**
     * Scroll to error message
     */
    scrollToError() {
        this.errorAlert.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }

    /**
     * Scroll to results section
     */
    scrollToResults() {
        this.resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }
}

/**
 * Global functions for button actions
 */
function tryAnother() {
    const app = window.colorizationApp;
    app.hideResults();
    app.hideError();
    app.clearFileInput();
    
    // Scroll back to upload section
    document.querySelector('.upload-section').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

function downloadResults() {
    const images = [
        { id: 'originalImage', name: 'original.png' },
        { id: 'grayscaleImage', name: 'grayscale.png' },
        { id: 'colorizedImage', name: 'ai_colorized.png' }
    ];

    images.forEach(img => {
        const element = document.getElementById(img.id);
        if (element && element.src) {
            downloadImage(element.src, img.name);
        }
    });
}

function downloadImage(dataUrl, filename) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Utility functions
 */
function showToast(message, type = 'info') {
    // Simple toast notification
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 10000; min-width: 300px;';
    toast.innerHTML = `
        <i class="fas fa-info-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

/**
 * Initialize app when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the app
    window.colorizationApp = new ImageColorizationApp();
    
    // Add smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation to images
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('load', function() {
            this.classList.add('fade-in-up');
        });
    });
    
    console.log('Image Colorization App initialized successfully!');
});

/**
 * Handle window resize
 */
window.addEventListener('resize', function() {
    // Handle any responsive adjustments if needed
});

/**
 * Handle network status
 */
window.addEventListener('online', function() {
    showToast('Connection restored!', 'success');
});

window.addEventListener('offline', function() {
    showToast('Connection lost. Please check your internet connection.', 'warning');
});