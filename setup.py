from setuptools import setup, find_packages

setup(
    name='image_text_grounding',
    version='0.1.0',
    description='An image-text grounding model for medical images and reports',
    author='Arnold Caleb Asiimwe',
    author_email='aa4870@columbia.edu',
    url='https://github.com/arnold-caleb/image-text-grounding.git',
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
