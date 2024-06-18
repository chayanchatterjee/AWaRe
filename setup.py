from setuptools import setup

dependencies=[
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

setup(
    name='AWaRe',
    version='0.1',
    description='Package for AWaRe - a deep learning model for gravitational wave signal reconstruction',
    url='https://github.com/chayanchatterjee/AWaRe',
    python_requires='>=3.8',
    packages=['evaluation', 'evaluation.configs', 'evaluation.model', 'evaluation.Plots', 'evaluation.Saved_results_files', 'evaluation.Test_data', 'evaluation.utils'],
    package_data={'evaluation':['Test_data/*.hdf']},
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'aware-evaluate=evaluation.evaluate:main'
        ],
    },
    author='Chayan Chatterjee',
    author_email='chayan.chatterjee@vanderbilt.edu',
)
