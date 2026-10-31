"""quadsim: quadrotor Sim-to-Real reinforcement-learning platform on Isaac Lab.

Sub-packages
  core    framework-agnostic PyTorch physics, controllers and sensor models (no Isaac import)
  isaac   Isaac Lab integration (spawners, scene helpers)
  tasks   Gymnasium-registered Isaac Lab environments
  rl      network architectures and training utilities
  data    synchronized data recording
  deploy  policy export and runtime interfaces (ROS 2 / PX4)
"""

__version__ = "0.1.0"
