from setuptools import find_packages, setup


setup(
    name="group-photo-optimizer",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.24.4",
        "opencv-contrib-python==4.10.0.84",
        "mediapipe==0.10.11",
        "Pillow==10.4.0",
        "PyYAML==6.0.2",
        "scipy==1.10.1",
        "tqdm==4.67.1",
        "pywebview==5.4",
    ],
    entry_points={
        "console_scripts": ["group-photo-optimize=group_photo_optimizer.cli:main"]
    },
)
