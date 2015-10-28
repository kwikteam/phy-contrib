# phy-contrib

[![Build Status](https://img.shields.io/travis/kwikteam/phy-contrib.svg)](https://travis-ci.org/kwikteam/phy-contrib)
[![codecov.io](https://img.shields.io/codecov/c/github/kwikteam/phy-contrib.svg)](http://codecov.io/github/kwikteam/phy-contrib?branch=master)
[![PyPI release](https://img.shields.io/pypi/v/phy-contrib.svg)](https://pypi.python.org/pypi/phy-contrib)

Community-contributed set of plugins for phy

## Easy install

```
$ pip install phycontrib
```

This will install `phycontrib` and create a `~/.phy/plugins/phycontrib_loader.py` file that will just import `phycontrib`. This will automatically load all phycontrib plugins.

## Advanced install

Just clone the repository in your `~/.phy/plugins/`. All Python files in that directory are automatically loaded, and the plugins defined there are readily available.

```
$ cd ~/.phy/plugins/
$ git clone https://github.com/kwikteam/phy-contrib
```

## Plugins

Every plugin is implemented in a subdirectory inside `phycontrib/`.

To implement a plugin, just create a Python script with a class deriving from `phy.IPlugin`:

```python
from phy import IPlugin

class MyPlugin(IPlugin):
    pass
```

There, you can implement

* `MyPlugin.attach_to_cli(cli)`: This is called when the `phy` command-line script is called. You have a chance to customize the script and add your own subcommands using the [**click** library](http://click.pocoo.org/5/).

* `MyPlugin.attach_to_gui(gui, ctx)`: This is called when you attach that plugin to a `GUI` instance. In that function, you can add views, create actions, connect to events, and do anything you want on the GUI. This allows you to create independent components for GUIs.
