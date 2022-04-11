from setuptools import setup, find_packages


setup(
    name='siampose',
    version='0.1.0',
    packages=find_packages(include=['siampose', 'siampose.*']),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'main=siampose.main:main'
        ],
    }
)
