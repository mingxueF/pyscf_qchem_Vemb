from setuptools import setup, find_packages

setup(
    name="pyscf_qchem_Vemb",  # Replace with the desired package name
    version="0.1.0",  # Update the version as needed
    author="mingxueF",  # Replace with your name or organization
    author_email="fmingxue@gmail.com",  # Replace with your email address
    description="A Python package for quantum chemistry with FDET Vemb integrations",  # Add a concise description
    long_description=open("README.md").read(),  # Ensure you have a README.md file
    long_description_content_type="text/markdown",
    url="https://github.com/mingxueF/pyscf_qchem_Vemb",  # GitHub repository URL
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify the minimum Python version
    install_requires=[
        # Add your package dependencies here
        "numpy",
        "scipy",
        "pyscf",
    ]
)
