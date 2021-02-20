import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as fp:
    install_requires = fp.read()

print(setuptools.find_packages())

setuptools.setup(
    name="heartDiseasePrediction",
    version="1.0",
    description="Heart Disease Prediction",
    author="Toby Wilkinson",
    author_email="NA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: python :: 3",
        "Licence :: OSI Approved :: MIT Licence",
        "Operating System :: OS Independent"
    ],
    python_requires=">3.7",
)