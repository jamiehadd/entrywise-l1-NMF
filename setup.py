from setuptools import setup

setup(
        name='entrywise-l1-NMF',
        version='0.1',
        description="Entrywise L1-Norm Non-Negative Matrix Factorization",
        url='https://github.com/jamiehadd/entrywise-l1-NMF',
        author='Toby Anderson, Jamie Haddock, Alicia Lu',
        author_email='allu@g.hmc.edu, tobanderson@g.hmc.edu',
        license='MIT',
        packages=[
            'entrywise-l1-NMF'
        ],
        install_requires=[
            'numpy'
        ]
)
