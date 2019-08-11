#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'https://github.com/Theano/Theano/archive/master.zip',
    'https://github.com/Lasagne/Lasagne/archive/master.zip',
]

tests_require = [
    'pytest',
]

setup(
    name='apes',
    version='0.0.1',
    author='Zhi Rui Tam',
    author_email='zhirui09400@icloud.com',
    license='MIT',
    packages=find_packages(),
    package_data={'apes': ['apes/*', 'qa_system/*']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    tests_require=tests_require,
    description='APES : a context focus summary evaluation metric',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)