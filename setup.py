from setuptools import setup
from setuptools import find_packages, setup


setup(
        name='entrywise-l1-NMF',
        version='0.1',
        description="Entrywise L1-Norm Non-Negative Matrix Factorization",
        url='https://github.com/jamiehadd/entrywise-l1-NMF',
        author='Toby Anderson, Jamie Haddock, Alicia Lu',
        author_email='allu@g.hmc.edu, tobanderson@g.hmc.edu',
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        license='MIT',
        install_requires=[
            'numpy'
        ]
)
