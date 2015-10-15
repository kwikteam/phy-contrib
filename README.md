# phy-contrib
Community-contributed set of plugins for phy

## Easy install

```
$ pip install phy-contrib
```

This will install `phy-contrib` and create a symlink from your `~/.phy/plugins` to your installed `phy-contrib` package.

## Advanced install

Just clone the repository in your `~/.phy/plugins/`. All Python files in that directory are automatically loaded, and the plugins defined there are readily available.

```
$ cd ~/.phy/plugins/
$ git clone https://github.com/kwikteam/phy-contrib
```

## Plugins

This repository contains a `plugins/` folder. Here, every plugin is implemented in a subdirectory. The name of the subdirectory is the name of the plugin. For example, the `kwik` plugin is in `plugins/kwik/`.

To implement a plugin, just create a Python script with a class deriving from `phy.IPlugin`:

```python
from phy import IPlugin

class MyPlugin(IPlugin):
    pass
```

Currently, you can implement two methods in your plugin:

* `MyPlugin.attach_to_cli(cli)`: This is called when the `phy` command-line script is called. You have a chance to customize the script and add your own subcommands using the [**click** library](http://click.pocoo.org/5/).

* `MyPlugin.attach_to_gui(gui)`: This is called when you attach that plugin to a `GUI` instance (i.e. when you do `gui.attach('MyPlugin')`. In that function, you can add views, create actions, and do anything you want on the GUI. This allows you to create independent components for GUIs.

By implementing two methods, you can create a custom subcommand `phy mysubcommand arg1 --option1=value1` that launches a custom GUI.
