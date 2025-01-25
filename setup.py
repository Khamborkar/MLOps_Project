from setuptools import setup, find_packages

setup(
    name='MLOps_Project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow',
        'pytest',
        'flake8',
        'scikit-learn',
        'pandas',
        # Add any other dependencies you require
    ],
)
