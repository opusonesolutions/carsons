import os.path
from setuptools import setup

thisdir = os.path.abspath(os.path.dirname(__file__))
version = open(os.path.join(thisdir, 'carsons', 'VERSION')).read().strip()


def readme():
    with open("README.md", 'r', encoding='UTF-8') as f:
        return f.read()


setup(
    name="carsons",
    version=version,
    packages=["carsons"],
    package_data={
        '': ['VERSION'],
        'carsons': ['py.typed'],
    },
    description="A python library computing carson's equations.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    long_description=readme(),
    long_description_content_type='text/markdown',
    author="Opus One Solutions",
    author_email="rnd@opusonesolutions.com",
    url="https://github.com/opusonesolutions/carsons",
    keywords=["carsons", "cables", "lines", "power systems"],
    license="MIT",
    install_requires=[
        'numpy>=1.13.1',
    ],
    zip_safe=False,
    extras_require={
        "test": [
            "flake8",
            "coveralls",
            "pytest>=3.6",
            "pytest-cov",
            "pytest-mypy",
            "pint",
        ],
    },
)
