# Customer Churn Prediction System

A comprehensive web application that predicts customer churn probability using machine learning and provides actionable insights through an intuitive user interface.

## 🌟 Features

- **Real-time Churn Prediction**: Predict customer churn probability using a trained machine learning model
- **Interactive Web Interface**: User-friendly form for inputting customer data
- **Detailed Analysis**: Comprehensive breakdown of prediction results with risk levels
- **Responsive Design**: Works seamlessly on both desktop and mobile devices
- **Data Visualization**: Visual representation of prediction results and risk factors

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Processing**: Pandas, NumPy
- **Visualization**: Chart.js

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Flask application:
```bash
python src/app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the prediction form to:
   - Enter customer information
   - Select service details
   - Input payment information
   - Get instant churn predictions

## 📁 Project Structure

```
customer-churn-prediction/
├── data/                  # Dataset and data processing scripts
├── models/               # Trained machine learning models
├── src/                  # Source code
│   ├── static/          # Static files (CSS, JS, images)
│   ├── templates/       # HTML templates
│   ├── utils/           # Utility functions
│   └── app.py           # Main Flask application
├── venv/                # Virtual environment
├── .gitignore          # Git ignore file
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Configuration

The application can be configured through environment variables:

- `FLASK_ENV`: Set to 'development' or 'production'
- `FLASK_DEBUG`: Enable/disable debug mode
- `MODEL_PATH`: Path to the trained model file

## 📊 Model Details

The prediction system uses a Random Forest Classifier trained on historical customer data. The model considers various features including:

- Customer demographics
- Service usage patterns
- Contract details
- Payment information
- Technical support usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset providers
- Open-source community
- Contributors and maintainers

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🔄 Updates and Maintenance

The project is actively maintained. Check the repository for the latest updates and improvements. 