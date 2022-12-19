from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

import OISST

setup(name='OISST',
      description='code for calculation and mapping of SST anomalies and heatwaves conditions around Aotearoa / New Zealand from daily OISST',
      url='https://github.com/nicolasfauchereau/OISST_indices',
      author='Nicolas Fauchereau',
      author_email='Nicolas.Fauchereau@gmail.com',
      license='MIT',
      packages=find_packages(exclude=["tests", "docs"]),
      version=OISST.__version__, 
      zip_safe=False)
