# phy-contrib

[![Build Status](https://img.shields.io/travis/kwikteam/phy-contrib.svg)](https://travis-ci.org/kwikteam/phy-contrib)
[![PyPI release](https://img.shields.io/pypi/v/phycontrib.svg)](https://pypi.python.org/pypi/phycontrib)

Plugins for [**phy**](https://github.com/kwikteam/phy). Currently, this package provides two integrated spike sorting GUIs:

* **KwikGUI**: to be used with Klusta and the Kwik format (successor of KlustaViewa)
* **TemplateGUI**: to be used with KiloSort and SpykingCircus

## Quick install

You first need to install [**phy**](https://github.com/kwikteam/phy). Then, activate your conda environment and do:

```bash
pip install phycontrib
```

### Installing the development version

If you want to use the bleeding-edge version, do:

```
git clone https://github.com/kwikteam/phy-contrib.git
cd phy-contrib
python setup.py develop
```

Then, you can update at any time with:

```
cd phy-contrib
git pull
```

## Documentation

* [Main documentation](http://phy-contrib.readthedocs.io/en/latest/)
* KwikGUI documentation (work in progress)
* [TemplateGUI documentation](http://phy-contrib.readthedocs.io/en/latest/template-gui/)
