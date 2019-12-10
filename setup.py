import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
  name = 'autodiff-kgl',
  packages=setuptools.find_packages(), # packages = ['autodiff'],
  version = '0.3.3',
  license = 'MIT',
  description = 'Implementation of automatic differentiation using forward mode',
  long_description = long_description,
  long_description_content_type = "text/markdown",
  author = 'Jade Kessler, Hardik Gupta, and Cooper Lorsung',
  author_email = 'hardikgupta@g.harvard.edu',
  url = 'https://github.com/make-AD-ifference/cs207-FinalProject',
  classifiers = [
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    "Operating System :: OS Independent",
  ],
  python_requires = '>=3.6',
  install_requires = ['numpy','pytest','pytest-cov','matplotlib'],
)