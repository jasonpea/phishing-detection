document.getElementById('checkBtn').addEventListener('click', function() {
    const url = document.getElementById('url').value;

    if (!url) {
        document.getElementById('result').textContent = 'Please enter a URL';
        return;
    }

    // Extract features from the URL (replaces the Python extract_features function)
    const urlFeatures = extractFeatures(url);

    // Check if the features are valid (you could add more validations here)
    if (!urlFeatures || urlFeatures.length === 0) {
        document.getElementById('result').textContent = 'Could not extract features from the URL';
        return;
    }

    // Make a POST request to your Flask server
    fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: urlFeatures })
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction === 1) {
            document.getElementById('result').textContent = 'Phishing URL!';
        } else {
            document.getElementById('result').textContent = 'Legitimate URL.';
        }
    })
    .catch(error => {
        document.getElementById('result').textContent = 'Error: ' + error;
    });
});

// This function mimics your Python-based URL feature extraction in JavaScript
function extractFeatures(url) {
    const features = [];

    // Example features, based on what your Python version was doing
    features.push(url.length);  // Length of the URL
    features.push((url.match(/\./g) || []).length);  // Count dots in the URL
    features.push((url.split('.').length - 2));  // Number of subdomains
    features.push((url.match(/\//g) || []).length - 2);  // Count slashes
    features.push(url.startsWith('https') ? 1 : 0);  // HTTPS flag
    features.push(url.includes('@') ? 1 : 0);  // '@' in the URL
    features.push(url.includes('-') ? 1 : 0);  // '-' in the URL
    features.push((url.match(/\d/g) || []).length);  // Count of digits
    features.push((url.match(/[A-Z]/g) || []).length);  // Count uppercase letters
    features.push((url.match(/[a-z]/g) || []).length);  // Count lowercase letters
    features.push((url.match(/[%]/g) || []).length);  // Count percentage symbols
    features.push((url.match(/[_]/g) || []).length);  // Count underscores

    // Add more custom features based on your dataset
    features.push(url.includes('login') ? 1 : 0);  // Suspicious keyword
    features.push((url.match(/~/g) || []).length);  // Count tildes
    features.push(url.includes('confirm') ? 1 : 0);  // Another suspicious keyword

    // Adjust or remove features that are not applicable in JS but relevant in Python

    return features;
}
