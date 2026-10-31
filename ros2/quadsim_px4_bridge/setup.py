from setuptools import setup

setup(
    name="quadsim_px4_bridge",
    version="0.1.0",
    packages=["quadsim_px4_bridge"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/quadsim_px4_bridge"]),
        ("share/quadsim_px4_bridge", ["package.xml"]),
        ("share/quadsim_px4_bridge/launch", ["launch/policy.launch.py"]),
        ("share/quadsim_px4_bridge/config", ["config/hover.yaml"]),
    ],
    install_requires=["setuptools", "numpy", "onnxruntime"],
    entry_points={"console_scripts": ["policy_node = quadsim_px4_bridge.policy_node:main"]},
)
