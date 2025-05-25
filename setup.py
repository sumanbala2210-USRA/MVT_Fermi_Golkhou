from setuptools import setup, find_packages

# Extract metadata from the __init__.py
about = {}
with open("MVTfermi/__init__.py") as f:
    exec(f.read(), about)

setup(
    name='mvtfermitools',
    version=about['__version__'],
    author=about['__authors__'],
    author_email=about['__email__'],
    description="A Python package for Minimum Variability Timescale (MVT) analysis, tailored for Fermi GBM data.",
    long_description=about['__description__'],
    long_description_content_type="text/plain",
    packages=find_packages(include=["MVTfermi", "MVTfermi.*"]),
    include_package_data=True,
    python_requires='>=3.11',
    install_requires=[
        'astro-gdt @ git+https://github.com/USRA-STI/gdt-core.git',
        'astro-gdt-fermi @ git+https://github.com/USRA-STI/gdt-fermi.git',
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "pandas",
        "PyPDF2>=3.0.0",
        "termcolor",
        "sigfig",
        "requests",
        "jupyter",
        "notebook"
    ],
    entry_points={
        'console_scripts': [
            'MVTfermi = MVTfermi.mvt_parallel_fermi:main'
        ]
    },
    license=about['__license__'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: Apache Software License",
    ],
)
