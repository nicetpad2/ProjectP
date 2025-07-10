# NICEGOLD ENTERPRISE - Python Requirements Explanation

This project requires a set of core and advanced Python libraries to ensure full production, enterprise-grade, and AI-powered functionality. All dependencies are pinned to tested versions for maximum reliability and reproducibility. Below is a summary of the main libraries and their roles:

## Core Data Science
- **numpy**: Numerical computing
- **pandas**: Data analysis and manipulation
- **scikit-learn**: Machine learning algorithms and utilities
- **scipy**: Scientific computing
- **joblib**: Model serialization and parallel processing

## Machine Learning / Deep Learning
- **tensorflow**: Deep learning (CNN-LSTM, etc.)
- **torch**: Deep learning alternative (DQN, etc.)
- **stable-baselines3**: Reinforcement learning algorithms
- **gymnasium**: RL environment toolkit

## Feature Selection & Optimization
- **shap**: Feature importance analysis
- **optuna**: Hyperparameter optimization

## Data Processing & Configuration
- **PyYAML**: Configuration file parsing
- **PyWavelets**: Wavelet analysis
- **imbalanced-learn**: Handling class imbalance
- **ta**: Technical analysis indicators
- **opencv-python-headless**: Computer vision (headless)
- **Pillow**: Image processing

## Visualization
- **matplotlib, seaborn, plotly**: Data visualization and interactive charts

## Development Tools
- **pytest, black, flake8**: Testing, code formatting, linting

---

**All packages are required for full production and dashboard features.**
- Do not comment out any core ML/AI libraries for production use.
- For custom/enterprise modules, ensure all dependencies are listed in `requirements.txt`.
- To install: `pip install -r requirements.txt`

---

For more details, see the comments in `requirements.txt`.
