
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="DiscreteHillClimbing", 
    version="1.1.0",
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer = 'Demetry Pascal',
    description="An easy python implementation of Hill Climbing algorithm for tasks with discrete variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/DiscreteHillClimbing",
    keywords=['solve', 'optimization', 'problem', 'fast', 'combinatorial', 'easy', 'discrete'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['numpy']
    
    )





