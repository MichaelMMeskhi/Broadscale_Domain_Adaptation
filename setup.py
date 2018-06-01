import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bsda",
    version="0.0.1",
    author="Mikhail M.Meskhi",
    author_email="m.meskhi@na.edu",
    description="Experiment package for Broadscale Domain Adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelMMeskhi/Broadscale_Domain_Adaptation",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)