from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="landscapy",
    version="0.1.0",  # Updated version to reflect new features
    author="Fitness Landscape Team",
    author_email="example@example.com",
    description="A package for analyzing fitness landscapes modeled as network graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RSCJacksonLab/landscapy",
    project_urls={
        "Bug Tracker": "https://github.com/RSCJacksonLab/landscapy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "torch>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "flake8>=3.8",
            "black>=20.8b1",
        ],
    },
)
