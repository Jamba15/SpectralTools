from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spectraltools",
    version="0.3.0",
    author="Lorenzo Giambagli",
    description="Tools for train and prune in the dual space a fully connected layer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['spectraltools'],
    # install_requires=["tensorflow-gpu"],
    include_pachage_data=True,
    python_requires=">=3.7"
)
