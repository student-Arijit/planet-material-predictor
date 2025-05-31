# Planet Material Predictor 🚀

A comprehensive Streamlit web application for analyzing and predicting Mars material components using seismic and geological data.

## 🌟 Features

- **Data Analysis**: Comprehensive statistical analysis of Mars geological data
- **Interactive Visualizations**: Dynamic charts and plots using Plotly and Seaborn
- **Material Prediction**: Machine learning models to predict material components
- **Multi-file Processing**: Handle multiple Excel files with seismic data
- **Real-time Statistics**: Live calculations and data insights
- **User-friendly Interface**: Clean, intuitive Streamlit interface

## 📊 Data Overview

The application processes seismic network data containing:
- Network stations and locations
- Channel information
- Time series data
- Velocity measurements
- Calibration data
- Delta calculations

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/planet-material-predictor.git
cd planet-material-predictor
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run app/main.py
```

2. **Or use the console command:**
```bash
mars-predictor
```

3. **Open your browser** and navigate to `http://localhost:8501`

### Data Preparation

1. Place your Excel files in the `data/raw/` directory
2. Ensure files follow the expected format (Network, Station, Location, etc.)
3. The app will automatically detect and process all files

## 📁 Project Structure

```
planet-material-predictor/
│
├── data/
│   ├── raw/               # Raw Excel data files
│   ├── processed/         # Processed data files
│
├── models/
│   ├── trained_model.pkl  # Trained ML models
│   └── model_utils.py     # Model utilities
│
├── app/
│   ├── pages/           
│   │   ├── home.py        # Home page
│   │   └── prediction.py  # Prediction page
│   ├── components/        
│   │   └── custom_widgets.py  # Custom UI components
│   └── main.py            # Main application
│
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── setup.py              # Package setup
```

## 📈 Features Details

### Data Analysis
- **Statistical Summary**: Mean, median, std deviation for all numerical columns
- **Distribution Analysis**: Histograms and box plots
- **Correlation Analysis**: Heatmaps showing relationships between variables
- **Time Series Analysis**: Temporal patterns in velocity and other measurements

### Visualizations
- **Interactive Charts**: Plotly-based dynamic visualizations
- **Station Mapping**: Geographic distribution of measurement stations
- **Velocity Profiles**: Time-based velocity analysis
- **Material Distribution**: Component breakdown charts

### Machine Learning
- **Predictive Models**: Trained models for material component prediction
- **Feature Importance**: Analysis of key factors affecting predictions
- **Model Performance**: Accuracy metrics and validation results

## 🔧 Configuration

### Data Format Requirements
Your Excel files should contain these columns:
- `Network`: Network identifier
- `Station`: Station code
- `Location`: Geographic location
- `Channel`: Data channel
- `Start Time`: Measurement start time
- `End Time`: Measurement end time
- `Sampling`: Sampling rate
- `Delta`: Time delta
- `Number of Points`: Data points count
- `Calibration`: Calibration factor
- `Time`: Timestamp
- `Velocity`: Velocity measurements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Arijit Chowdhury**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Mars seismic data providers
- Streamlit community
- Scientific Python ecosystem contributors

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/planet-material-predictor/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your environment and the issue

## 🔄 Version History

- **v1.0.0** - Initial release with basic functionality
  - Data loading and processing
  - Statistical analysis
  - Basic visualizations
  - Material prediction capabilities

---

⭐ **Star this repository if you find it helpful!** ⭐