from setuptools import find_packages, setup

setup(
    name="g2maf",
    description="G²MAF: test-time gradient refinement for frozen multi-agent policies.",
    author="Anonymous Authors",
    url="",
    project_urls={
    },
    packages=find_packages(include=["diffuser", "diffuser.*", "g2maf", "g2maf.*"]),
)
