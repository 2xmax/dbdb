from setuptools import setup

setup(
    name="dbdb",
    version="0.0.1",
    py_modules=["dbdb", "ridge_nnls", "bet"],
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=1.2.1"
    ],
)
