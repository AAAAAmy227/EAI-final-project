import sapien
import sapien.physx as physx

scene = sapien.Scene()
# In SAPIEN 3, actors are entities with components
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[1,1,1])
actor = builder.build_static(name="test")

print(f"Actor: {actor}")
for comp in actor.components:
    print(f"Component: {comp}")
    if isinstance(comp, physx.PhysxRigidStaticComponent):
        for shape in comp.collision_shapes:
            mat = shape.physical_material
            print(f"Static Friction: {mat.static_friction}")
            print(f"Dynamic Friction: {mat.dynamic_friction}")
            print(f"Restitution: {mat.restitution}")
