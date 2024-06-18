from setuptools import setup, find_packages

setup(
    name='AWaRe',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.22.2',
        'h5py>=3.10.0',
        'tensorflow==2.8.0',
        'tensorflow-probability==0.16.0',
        'astropy==5.2.2',
        'seaborn>=0.13.2',
        'scipy==1.6.3',  
        'pycbc==2.4.0',
        'matplotlib>=3.6.2', 
        'matplotlib-latex-bridge==0.2.0'
    ],
    entry_points={
        'console_scripts': [
            'aware-evaluate=AWaRe.evaluation.evaluate:main',
            'aware-main=AWaRe.main:main'
        ],
    },
    author='Chayan Chatterjee',
    author_email='your_email@domain.com',
    description='Attention-boosted Waveform Reconstruction network for gravitational waves',
    url='https://github.com/chayanchatterjee/AWaRe',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
