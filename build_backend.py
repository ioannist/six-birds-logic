"""Compatibility wrapper to provide editable build hooks across setuptools versions."""

from setuptools import build_meta as _build_meta


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_meta.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def build_sdist(sdist_directory, config_settings=None):
    return _build_meta.build_sdist(sdist_directory, config_settings=config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _build_meta.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings=config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    build_editable_impl = getattr(_build_meta, "build_editable", None)
    if build_editable_impl is not None:
        return build_editable_impl(
            wheel_directory,
            config_settings=config_settings,
            metadata_directory=metadata_directory,
        )
    return _build_meta.build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def get_requires_for_build_editable(config_settings=None):
    get_requires_impl = getattr(_build_meta, "get_requires_for_build_editable", None)
    if get_requires_impl is not None:
        return get_requires_impl(config_settings=config_settings)
    return _build_meta.get_requires_for_build_wheel(config_settings=config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    prepare_metadata_impl = getattr(_build_meta, "prepare_metadata_for_build_editable", None)
    if prepare_metadata_impl is not None:
        return prepare_metadata_impl(
            metadata_directory,
            config_settings=config_settings,
        )
    return _build_meta.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=config_settings,
    )
