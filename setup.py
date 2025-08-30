from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="landscapy",
    version="0.9.0",
    author="Matthew A. Spence",
    author_email="matthew.spence@anu.edu.au",
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
    python_requires=">=3.12",
    install_requires=[
        "softalign @ git+https://github.com/RSCJacksonLab/softalign.git",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "torch>=1.7.0",
        "transformers>=4.53.2",
        "cogent3==2025.7.10a10", 
        "piqtree>=0.6.1",
        "pydantic==2.11.7",
        "pytest>=8",
        "pytest-cov>=6",
        "faiss-cpu>=1.11",
        "ray>=2.48",
        "gudhi>=3.11",
        "pandas>=2.3",
        "torch_geometric>=2.6"],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-mock>=3.14",
            "pytest-cov>=2.10",
            "flake8>=3.8",
            "black>=20.8b1",
        ],
    },
)
