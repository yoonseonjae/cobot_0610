import os
from pathlib import Path
from isaacsim.core.utils.prims import define_prim


class EnvironmentLoader:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = base_dir

    def spawn_map(self):
        prim = define_prim("/World/Map", "Xform")
        asset_path = os.path.join(self.base_dir, "map", "c_1_default_map.usd")
        prim.GetReferences().AddReference(asset_path)
        print(f"[Environment] Map loaded: {asset_path}")

        # 소화기(빨간 큐브 대체) 스폰 — robot1 대기방 앞
        from omni.isaac.core.objects import FixedCuboid
        from omni.isaac.core.prims import RigidPrim
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import numpy as np

        cube_x = 10.7 - 0.87
        cube_y = 0.5
        cube_z = 0.48

        try:
            from omni.isaac.core.materials import PhysicsMaterial
            PhysicsMaterial(
                prim_path="/World/Physics_Materials/HighFriction",
                dynamic_friction=5.0,
                static_friction=5.0,
                restitution=0.0,
            )

            extinguisher_usd = os.path.join(
                self.base_dir, "map", "fire_extinguisher", "World0.usd"
            )
            add_reference_to_stage(extinguisher_usd, "/World/Cube")
            RigidPrim(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([cube_x, cube_y, cube_z - 0.05]),
                mass=0.5,
            )

            from pxr import UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            cube_prim = stage.GetPrimAtPath("/World/Cube")
            if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(cube_prim)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(cube_prim)
                mesh_api.CreateApproximationAttr("boundingCube")

            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([cube_x, cube_y, cube_z - 0.025 - 0.05]),
                scale=np.array([0.2, 0.2, 0.1]),
                color=np.array([0.4, 0.4, 0.4]),
            )
            print("[Environment] 소화기 + 테이블 스폰 완료")
        except Exception as e:
            print(f"[Environment] 소화기 스폰 실패: {e}")

    def spawn_people(self):
        from omni.isaac.core.prims import XFormPrim
        import numpy as np

        # Person1 (Biped_Setup)
        person1_prim = define_prim("/World/Person1", "Xform")
        person1_prim.GetReferences().AddReference(
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.1/Isaac/People/Characters/Biped_Setup.usd"
        )
        XFormPrim("/World/Person1").set_local_pose(
            translation=np.array([-4.67817, 3.28751, 0.15715]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        # Person2 (Female Police)
        person2_prim = define_prim("/World/Person2", "Xform")
        person2_prim.GetReferences().AddReference(
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
            "/Assets/Isaac/5.1/Isaac/People/Characters/female_adult_police_01_new"
            "/female_adult_police_01_new.usd"
        )
        try:
            from scipy.spatial.transform import Rotation as R
            r = R.from_euler("xyz", [-2.811, -2.486, 87.911], degrees=True).as_quat()
            orient = np.array([r[3], r[0], r[1], r[2]])
        except ImportError:
            orient = np.array([1.0, 0.0, 0.0, 0.0])
        XFormPrim("/World/Person2").set_local_pose(
            translation=np.array([-3.98723, 12.25463, 0.01957]),
            orientation=orient,
        )
        print("[Environment] Person1, Person2 스폰 완료")

    def apply_map_collisions(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        map_prim = stage.GetPrimAtPath("/World/Map")
        if map_prim:
            for p in Usd.PrimRange(map_prim):
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(p)
                    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(p)
                    mesh_api.CreateApproximationAttr("none")
            print("[Environment] 맵 충돌체 적용 완료")
