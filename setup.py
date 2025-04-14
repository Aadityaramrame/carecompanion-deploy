from setuptools import setup, find_packages

setup(
    name="carecompanion",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "pytesseract>=0.3.10",
        "opencv-python-headless>=4.5.3.56",
        "Pillow>=9.0.1",
        "numpy>=1.22.4",
        "transformers>=4.26.1",
        "torch>=1.13.1",
        "sentencepiece>=0.1.97",
        "spacy>=3.5.0",
        "scikit-learn>=1.2.2",
        "pandas>=2.0.1",
        "googletrans==4.0.0-rc1",
        "regex>=2023.5.5",
        "protobuf>=3.20.3",
        "tokenizers>=0.13.3",
        "gunicorn>=20.1.0",
        "python-dotenv>=0.20.0",
        "pydantic>=1.10.7"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "carecompanion=main:main"
        ]
    },
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "OCR": ["*.json", "*.txt"],
        "summarizer": ["*.json"],
        "Chatbot": ["*.json"]
    },
    author="anusri",
    author_email="your.email@example.com",
    description="CareCompanion API for medical document processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aadityarammate/carecompanion-deploy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
