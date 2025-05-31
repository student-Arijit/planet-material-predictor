from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="planet-material-predictor",
    version="1.0.0",
    author="Arijit Chowdhury",
    author_email="arijitchowdhury4467@gmail.com",
    description="A Streamlit web application for Mars material component statistics and prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/student-arijit/planet-material-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mars-predictor=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
        "data": ["raw/*", "processed/*"],
        "models": ["*.pkl", "*.joblib"],
    },
    keywords="mars, material, prediction, streamlit, seismic, geology, planetary science",
    project_urls={
        "Bug Reports": "https://github.com/student-arijit/planet-material-predictor/issues",
        "Source": "https://github.com/student-arijit/planet-material-predictor",
        "Documentation": "https://github.com/student-arijit/planet-material-predictor/wiki",
    },
)