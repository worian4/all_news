from setuptools import setup, find_packages

setup(
    name="news-aggregator-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "telethon>=1.28.5",
        "sentence-transformers>=2.2.2",
        "aiofiles>=23.2.1",
        "scikit-learn>=1.3.2",
        "numpy>=1.24.3",
    ],
    python_requires=">=3.8",
)