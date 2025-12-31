import gymnasium as gym
import torch
import numpy as np
import sapien
import sapien.physx as physx
from grasp_cube.envs.tasks.pick_cube_so101 import PickCubeSO101Env

env = gym.make("PickCubeSO101-v1")
cube = env.unwrapped.cube

print(f"Cube: {cube.name}")
# Get entities
for entity in cube._objs:
    print(f"Checking entity: {entity.name}")
    for comp in entity.components:
        if isinstance(comp, physx.PhysxRigidDynamicComponent):
            print(f"  RigidDynamicComponent: {comp.name}")
            for shape in comp.collision_shapes:
                mat = shape.physical_material
                print(f"    Static Friction: {mat.static_friction}")
                print(f"    Dynamic Friction: {mat.dynamic_friction}")
                print(f"    Restitution: {mat.restitution}")
            print(f"  Mass: {comp.mass}")

env.close()
