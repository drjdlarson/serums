import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="serums",
    version="0.0.0",
    description="A Python package for Statistical Error and Risk Utility for Multi-sensor Systems (SERUMS).",
    # long_description=readme(),
    url="https://github.com/drjdlarson/serums",
    author="Laboratory for Autonomy, GNC, and Estimation Research (LAGER)",
    author_email="",
    license="MIT",
    packages=setuptools.find_packages(),
    package_dir={},
    install_requires=["numpy", "scipy", "matplotlib", "probscale", 'pandas'],
    tests_require=["pytest", "numpy"],
    include_package_data=True,
    zip_safe=False,
)

