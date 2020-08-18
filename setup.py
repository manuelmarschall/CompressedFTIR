import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="compressedftir", # Replace with your username

    version="0.0.1",

    author="Manuel Marschall",

    author_email="manuelmarschall@ptb.de",

    description="Reconstruction of sample FTIR data",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="<https://github.com/manuelmarschall/CompressedFTIR>",

    # packages=setuptools.find_packages(),
    packages=["compressedftir"],

    classifiers=[

        "Programming Language :: Python :: 3",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)