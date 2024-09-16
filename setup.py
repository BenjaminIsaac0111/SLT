from setuptools import setup, find_packages

setup(
    name='Attention-UNET',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='GPT-3.0 License',
    author='Benjamin Wilson',
    author_email='benjamintaya0111@gmail.com',
    description='Attention U-Net built in Keras 2 and TensorFlow 2.',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GPT-3.0 License",
        "Operating System :: OS Independent",
    ],
    # NOTE: I have only tested these specific dependencies on Windows Systems. These are the last versions of TF for
    # on available on Windows 10/11 OS. New versions of TF > 2.10 must be installed via Windows Subsystem Linux 2 (WSL2)
    # or a linux system.
    install_requires=[
        "tensorflow~=2.10.0",
        "tensorflow-addons==0.20.0",
        "pyaml",
        "scikit-learn==1.3.0",

    ],
    python_requires='==3.8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
