from setuptools import setup, find_packages

setup(
    name="lumen-splat",
    version="0.1.0",
    description="Real-time 3D Gaussian Splatting and Radiance Field Rasterization Engine",
    author="Aneesh Krishna",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
    ],
)
