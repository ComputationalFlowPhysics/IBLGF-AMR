"""Use the available TOML reader across supported Python versions."""

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        try:
            import toml as _toml
        except ImportError as error:
            raise ImportError(
                "Reading vortex configuration files requires Python 3.11+, "
                "or the 'tomli' or 'toml' package."
            ) from error

        class _TomlCompat:
            @staticmethod
            def load(handle):
                contents = handle.read()
                if isinstance(contents, bytes):
                    contents = contents.decode("utf-8")
                return _toml.loads(contents)

        tomllib = _TomlCompat()
