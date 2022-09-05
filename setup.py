import sys
import setuptools

__version__ = '0.1'
__patch__ = '1'

VERSION             = '%s.%s' % (__version__, __patch__)
ISRELEASED          = False

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def load_requirements():
    requirements_file_name = "requirements.txt"
    requires = []
    with open(requirements_file_name) as f:
        for line in f:
            if line:
                requires.append(line.strip())
    return requires

setuptools.setup(
    name="mini-photoshop",
    version=VERSION,
    author="TungBui",
    author_email="tungbui198.hust@gmail.com",
    description="A free and opensource tool, which make some awesome things with your pictures!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages("./"),
    python_requires=">=3.8",
    entry_points={"console_scripts": ["mini-photoshop=mini_photoshop.gui:entry_point"]},
)
