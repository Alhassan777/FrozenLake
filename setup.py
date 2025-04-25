#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="frozen-lake-gui",
    version="1.0.0",
    description="A GUI for the Frozen Lake environment from Gymnasium",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pygame",
        "numpy",
        "matplotlib",
        "tqdm",
        "torch",
        "pandas",
        "seaborn",
        "ipython",
    ],
    entry_points={
        "console_scripts": [
            "frozen-lake-gui=run:main",
            "frozen-lake-auto-solver=run:main_auto_solver",
            "frozen-lake-agent-comparison=run_agent_comparison:main",
            "frozen-lake-agent-comparison-gui=run_agent_comparison_gui:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 