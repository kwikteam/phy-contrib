# -*- coding: utf-8 -*-

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

setup(
    name='phycontrib',
    version='1.0',
    license='BSD',
    description='phycontrib',
    long_description='',
    author='Kwik Team',
    author_email='cyrille.rossant at gmail.com',
    url='https://phy.cortexlab.net',
    packages=[
        'phycontrib',
        'phycontrib.kwik_gui',
        'phycontrib.template',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
