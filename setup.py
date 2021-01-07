from setuptools import setup, find_packages


setup(
    name="learn2branch",
    version="1.0.0",
    author="Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi",
    install_requires=["numpy", "scipy"],
    packages=["learn2branch"],
    package_dir={"learn2branch": "."},
)
