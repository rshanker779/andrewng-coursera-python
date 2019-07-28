from setuptools import setup, find_packages


setup(
    name="andrewng-coursera-python",
    version="1.0.0",
    author="rshanker779",
    author_email="rshanker779@gmail.com",
    description="Python answers to Andrew Ng's coursera Machine learning course",
    license="MIT",
    python_requires=">=3.5",
    install_requires=[
        "black",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    packages=find_packages(),
    entry_points={},
)
