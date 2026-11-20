// Add fade-in animation to cards
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.classList.add('fade-in');
    });
});

// Form validation
function validateForm(form) {
    if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
    }
    form.classList.add('was-validated');
}

// Add loading state to buttons
function setLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    } else {
        button.disabled = false;
        button.innerHTML = button.getAttribute('data-original-text');
    }
}

// Save button original text
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('button[type="submit"]');
    buttons.forEach(button => {
        button.setAttribute('data-original-text', button.innerHTML);
    });
});

// Handle form submissions
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                setLoading(submitButton, true);
            }
        });
    });
});

// Add smooth scrolling
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

// Handle responsive navigation
document.addEventListener('DOMContentLoaded', function() {
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        document.addEventListener('click', function(event) {
            const isClickInside = navbarToggler.contains(event.target) || 
                                navbarCollapse.contains(event.target);
            
            if (!isClickInside && navbarCollapse.classList.contains('show')) {
                navbarCollapse.classList.remove('show');
            }
        });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // Handle prediction form submission
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Validate form
            if (!predictionForm.checkValidity()) {
                e.stopPropagation();
                predictionForm.classList.add('was-validated');
                return;
            }

            // Collect form data and ensure all required fields are filled
            const getValue = (id) => {
                const el = document.getElementById(id);
                return el ? el.value : '';
            };
            const getChecked = (id) => {
                const el = document.getElementById(id);
                return el ? el.checked : false;
            };

            // Required select fields
            const requiredSelects = [
                'gender', 'multiple_lines', 'internet_service', 'online_security',
                'online_backup', 'device_protection', 'tech_support',
                'streaming_tv', 'streaming_movies', 'contract', 'payment_method'
            ];
            for (const id of requiredSelects) {
                if (!getValue(id)) {
                    alert('Please fill all required fields.');
                    return;
                }
            }

            // Required numeric fields
            const monthlyCharges = parseFloat(getValue('monthly_charges'));
            const tenure = parseInt(getValue('tenure'));
            if (isNaN(monthlyCharges) || isNaN(tenure)) {
                alert('Please enter valid numbers for Monthly Charges and Tenure.');
                return;
            }

            // Prepare form data for backend
            const formData = {
                gender: getValue('gender'),
                senior_citizen: getChecked('senior_citizen') ? 1 : 0,
                partner: getChecked('partner') ? 'Yes' : 'No',
                dependents: getChecked('dependents') ? 'Yes' : 'No',
                phone_service: getChecked('phone_service') ? 'Yes' : 'No',
                multiple_lines: getValue('multiple_lines'),
                internet_service: getValue('internet_service'),
                online_security: getValue('online_security'),
                online_backup: getValue('online_backup'),
                device_protection: getValue('device_protection'),
                tech_support: getValue('tech_support'),
                streaming_tv: getValue('streaming_tv'),
                streaming_movies: getValue('streaming_movies'),
                contract: getValue('contract'),
                paperless_billing: getChecked('paperless_billing') ? 'Yes' : 'No',
                payment_method: getValue('payment_method'),
                monthly_charges: monthlyCharges,
                tenure: tenure
            };

            // Show loading state
            const submitButton = predictionForm.querySelector('button[type="submit"]');
            const originalButtonText = submitButton.innerHTML;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            submitButton.disabled = true;

            // Send prediction request
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                // Hide loading indicator if present
                const loadingIndicator = document.querySelector('.loading-indicator, #loading-indicator');
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                // Create results card if it doesn't exist
                let resultsCard = document.getElementById('results-card');
                if (!resultsCard) {
                    resultsCard = document.createElement('div');
                    resultsCard.id = 'results-card';
                    resultsCard.className = 'card shadow mt-4 fade-in';
                    predictionForm.parentNode.insertBefore(resultsCard, predictionForm.nextSibling);
                }
                // Update results card content
                const riskClass = data.risk_level.toLowerCase();
                resultsCard.innerHTML = `
                    <div class="card-body">
                        <h3 class="card-title">
                            <i class="fas fa-chart-pie text-primary me-2"></i>
                            Prediction Results
                        </h3>
                        <div class="alert alert-${riskClass === 'high' ? 'danger' : riskClass === 'medium' ? 'warning' : 'success'}">
                            <h4 class="alert-heading">Churn Risk: ${data.risk_level}</h4>
                            <p class="mb-0">Probability of churn: ${data.churn_probability}%</p>
                        </div>
                        <div class="mt-3">
                            <h5>Recommendations:</h5>
                            <ul>
                                ${getRecommendations(data.risk_level)}
                            </ul>
                        </div>
                    </div>
                `;
                // Scroll to results
                resultsCard.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
                // Hide loading indicator if present
                const loadingIndicator = document.querySelector('.loading-indicator, #loading-indicator');
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                // Show error message
                let errorCard = document.getElementById('error-card');
                if (!errorCard) {
                    errorCard = document.createElement('div');
                    errorCard.id = 'error-card';
                    errorCard.className = 'alert alert-danger mt-4 fade-in';
                    predictionForm.parentNode.insertBefore(errorCard, predictionForm.nextSibling);
                }
                errorCard.innerHTML = `
                    <h4 class="alert-heading">Error</h4>
                    <p>${error.message || 'An error occurred while making the prediction. Please try again.'}</p>
                `;
                errorCard.scrollIntoView({ behavior: 'smooth' });
            })
            .finally(() => {
                // Reset button state
                submitButton.innerHTML = originalButtonText;
                submitButton.disabled = false;
            });
        });
    }
});

function getRecommendations(riskLevel) {
    const recommendations = {
        'High': [
            'Offer personalized retention incentives',
            'Schedule immediate customer service outreach',
            'Review and potentially adjust service pricing',
            'Provide additional value-added services'
        ],
        'Medium': [
            'Monitor customer behavior closely',
            'Send targeted promotional offers',
            'Check for service quality issues',
            'Consider loyalty program enrollment'
        ],
        'Low': [
            'Maintain regular engagement',
            'Continue providing excellent service',
            'Consider upselling opportunities',
            'Keep customer informed about new features'
        ]
    };

    return recommendations[riskLevel]
        .map(rec => `<li>${rec}</li>`)
        .join('');
} 