"""
Setup script for Image Colorization Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-colorization-vgg16",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered image colorization using VGG16 and custom CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-colorization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "gpu": [
            "torch[cuda]",
        ],
    },
    entry_points={
        "console_scripts": [
            "colorize-web=app:main",
            "colorize-train=training.train:main",
            "colorize-test=test_setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["frontend/templates/*.html", "frontend/static/**/*"],
    },
    keywords="deep-learning, image-colorization, vgg16, pytorch, computer-vision, ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/image-colorization/issues",
        "Source": "https://github.com/yourusername/image-colorization",
        "Documentation": "https://github.com/yourusername/image-colorization#readme",
    },
)