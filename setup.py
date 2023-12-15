from setuptools import setup, find_packages

setup(
    name='health-autocorrect',
    version='0.1.0',
    description='Image-Conditioned Autocorrection in medical reporting',
    author='Arnold Caleb Asiimwe',
    author_email='aa4870@columbia.edu',
    url='https://github.com/arnold-caleb/health-autocorrect',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'timm>=0.4.12',
        'transformers>=4.10.3',
        'numpy>=1.21.2',
        'scikit-learn>=0.24.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
