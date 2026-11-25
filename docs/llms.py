"""Prints the llms.txt of the documentation: the module map with the summaries of all public names."""

import inspect
import pkgutil
from importlib import import_module
from importlib.metadata import metadata
from types import ModuleType

import exar

URL = "https://wi2trier.github.io/exar"


def summary(obj: object) -> str:
    """The first line of the docstring of a module, class or function, empty for other objects."""
    if not (inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj)):
        return ""

    return (inspect.getdoc(obj) or "").split("\n", 1)[0]


def modules() -> list[ModuleType]:
    """The public submodules of exar that import, packages before their submodules."""
    names = sorted(
        (
            info.name
            for info in pkgutil.walk_packages(exar.__path__, "exar.")
            if not any(part.startswith("_") for part in info.name.split("."))
        ),
        key=lambda name: (name.count("."), name),
    )
    found: list[ModuleType] = []

    for name in names:
        try:
            found.append(import_module(name))
        except ImportError:
            continue

    return found


def main() -> None:
    """Prints every public name once, under the first module that exports it."""
    seen: set[int] = set()
    print(f"# EXAR\n\n> {metadata('exar')['Summary']}")
    print(
        "\nEvery module documents its details with tested examples,"
        " and every name has one public path, the export of its package."
    )

    for module in modules():
        page = f"{URL}/{module.__name__.replace('.', '/')}.html"
        entries = []

        for name in getattr(module, "__all__", []):
            obj = getattr(module, name, None)

            if obj is None or inspect.ismodule(obj) or id(obj) in seen:
                continue

            seen.add(id(obj))
            text = summary(obj)
            entries.append(f"- [{name}]({page}#{name})" + (f": {text}" if text else ""))

        print(f"\n## [{module.__name__}]({page})\n\n{summary(module)}")

        if entries:
            print("", *entries, sep="\n")


if __name__ == "__main__":
    main()
