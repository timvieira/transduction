from setuptools import setup

requirements = [
    'numpy',
    'graphviz',
    'IPython',
    'torch',
    'transformers',
    'matplotlib',
    'datasets',
    'tqdm',
    'pandas',
    'nltk',
    'pynini',
]


setup(
    name='transduction',
    version='0.1',
    description='',
    install_requires=requirements,
    python_requires='>=3.10',
    authors=[
        'Tim Vieira',
    ],
    readme='',
    scripts=[],
    packages=['transduction', 'transduction.applications', 'transduction.lm'],
)
