import numpy as np
import os
import json
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
import omni.kit.app
from pxr import Gf, UsdLux, Sdf
from spot_policy import SpotArmFlatTerrainPolicy


class SpotAgent:
    """
    robot1 (allow_grasp_trigger=True) : 소화기 Grasp + 투척 + Nav2
    robot2 (allow_grasp_trigger=False): Nav2 순찰 전용 (YOLO는 Stage 4)
    """

    def __init__(
        self,
        base_dir,
        enable_replicator_writer=False,
        namespace="robot1",
        spawn_pos=np.array([10.7, 0.5, 0.8]),
        udp_port=9876,
        allow_grasp_trigger=False,
    ):
        self.enable_replicator_writer = enable_replicator_writer
        self.namespace = namespace
        self.udp_port = udp_port
        self.allow_grasp_trigger = allow_grasp_trigger

        walking_policy_path = os.path.join(
            base_dir, "policies/spot_arm/models", "spot_arm_policy.pt"
        )
        balance_policy_path = os.path.join(
            base_dir, "policies/spot_arm/models", "model_10800.pt"
        )
        arm_balance_policy_path = None  # 76-dim 모델 없음
        policy_params_path = os.path.join(
            base_dir, "policies/spot_arm/params", "env.yaml"
        )
        usd_path = os.path.join(base_dir, "assets", "spot_arm.usd")

        self._spot = SpotArmFlatTerrainPolicy(
            prim_path=f"/World/{self.namespace}",
            name=self.namespace,
            usd_path=usd_path,
            walking_policy_path=walking_policy_path,
            balance_policy_path=balance_policy_path,
            arm_balance_policy_path=arm_balance_policy_path,
            policy_params_path=policy_params_path,
            position=spawn_pos,
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )

        self.first_step = True
        self._nav_command = np.zeros(3)
        self._pose_file = f"/tmp/isaac_pose_{self.namespace}.json"

        import socket
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(("127.0.0.1", self.udp_port))
        self.udp_sock.setblocking(False)

        # Grasp 상태머신 변수 (robot1 전용)
        self.ARM_IDX = [1, 0, 2, 7, 12, 17]
        self.GRIP_IDX = 18
        self.GRIP_CLOSE = 0.0
        self._delivery_state = "SEARCHING"
        self._is_heavy_mode = False
        self._carry_arm = None
        self._grasp_t = 0.0
        self._grabbed_cube_path = None
        self._is_thrown = False
        self._has_object = False

        # ---- robot1 자동 시나리오 변수 ----
        self._auto_fire_triggered = False
        # IDLE → NAV_TO_EXTINGUISHER → GRASPING → BACKUP → NAV_TO_FIRE → THROWING → DONE
        self._auto_state  = "IDLE"
        self._nav_wait_t  = 0.0
        self._backup_t    = 0.0   # BACKUP 상태 타이머
        self._nav_idle_t  = 0.0   # Nav2 cmd_vel 없을 때 직접 이동 폴백 타이머
        # 맵 좌표
        self.EXTINGUISHER_POS = (9.83, 0.5)     # 소화기
        self.FIRE_POS         = (0.158, -4.084)  # 2번방 화재

        # ---- robot2 순찰 변수 ----
        # 5→1→2→3→4→6→화장실
        self._patrol_waypoints = [
            (5.039,  -6.990),   # 5번방
            (-1.707, -11.104),  # 1번방
            (0.158,  -4.084),   # 2번방
            (-1.707,  2.059),   # 3번방
            (-1.707,  12.753),  # 4번방
            (5.039,   11.986),  # 6번방
            (3.668,   0.742),   # 화장실
        ]
        self._patrol_idx      = 0
        self._patrol_wait_t   = 0.0
        self._patrol_active   = False  # 화재 발생(10s) 시 True

        self._appwindow = omni.appwindow.get_default_app_window()
        import carb
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_agent_keyboard
        )

    # ------------------------------------------------------------------ #
    # Nav2 목표 전송 / 위치 헬퍼
    # ------------------------------------------------------------------ #
    def _send_nav2_goal(self, x, y, yaw=0.0):
        import subprocess, math
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal = (
            f"{{pose: {{header: {{frame_id: 'map'}}, "
            f"pose: {{position: {{x: {x}, y: {y}, z: 0.0}}, "
            f"orientation: {{x: 0.0, y: 0.0, z: {qz:.4f}, w: {qw:.4f}}}}}}}}}"
        )
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = ":".join(
                p for p in env["PYTHONPATH"].split(":") if "isaacsim" not in p.lower()
            )
        subprocess.Popen(
            ["ros2", "action", "send_goal",
             f"/{self.namespace}/navigate_to_pose",
             "nav2_msgs/action/NavigateToPose", goal],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[{self.namespace}] Nav2 목표: ({x:.2f}, {y:.2f})")

    def _get_robot_xy(self):
        try:
            pos, _ = self._spot.robot.get_world_pose()
            return float(pos[0]), float(pos[1])
        except Exception:
            return None, None

    def _dist_to(self, tx, ty):
        rx, ry = self._get_robot_xy()
        if rx is None:
            return 999.0
        return ((rx - tx) ** 2 + (ry - ty) ** 2) ** 0.5

    def _navigate_toward(self, tx, ty, max_linear=0.6, turn_gain=1.2):
        """Nav2 없이 Isaac Sim 물리엔진으로 직접 이동 — _nav_command를 매 스텝 갱신"""
        import math
        rx, ry = self._get_robot_xy()
        if rx is None:
            return
        dx = tx - rx
        dy = ty - ry
        target_yaw = math.atan2(dy, dx)
        try:
            _, q = self._spot.robot.get_world_pose()
            # q = [w, x, y, z]
            robot_yaw = math.atan2(
                2.0 * (q[0] * q[3] + q[1] * q[2]),
                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
            )
        except Exception:
            robot_yaw = 0.0
        err = target_yaw - robot_yaw
        while err > math.pi:  err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi
        # 정면에 가까울수록 전진, 많이 틀어지면 제자리 회전
        vx = max_linear * max(0.0, math.cos(err))
        wz = float(np.clip(turn_gain * err, -1.5, 1.5))
        self._nav_command[0] = vx
        self._nav_command[1] = 0.0
        self._nav_command[2] = wz

    def _stop_nav(self):
        self._nav_command[:] = 0

    # ------------------------------------------------------------------ #
    # robot1 자동 시나리오
    # ------------------------------------------------------------------ #
    def set_fire_detected(self):
        """main_simulation.py 화재 점화 시 호출 (robot1/robot2 공통)"""
        if not self._auto_fire_triggered:
            self._auto_fire_triggered = True
            if self.allow_grasp_trigger:
                print(f"[{self.namespace}] 화재 신호 수신 → 자동 소화 시나리오 시작")
            else:
                # robot2: 순찰 시작 + 첫 웨이포인트 Nav2 goal 전송
                self._patrol_active = True
                tx, ty = self._patrol_waypoints[0]
                self._send_nav2_goal(tx, ty)
                print(f"[{self.namespace}] 화재 신호 수신 → 순찰 시작 웨이포인트0 ({tx:.2f}, {ty:.2f})")

    def _run_auto_scenario(self, step_size):
        if not self._auto_fire_triggered:
            return

        if self._auto_state == "IDLE":
            print(f"\n[{self.namespace}] 🔥 화재 감지! 소화기로 Nav2 이동\n")
            self._send_nav2_goal(*self.EXTINGUISHER_POS)
            self._auto_state = "NAV_TO_EXTINGUISHER"
            self._nav_wait_t = 0.0

        elif self._auto_state == "NAV_TO_EXTINGUISHER":
            self._nav_wait_t += step_size
            dist = self._dist_to(*self.EXTINGUISHER_POS)
            if dist < 0.9:
                print(f"[{self.namespace}] 소화기 도착 ({dist:.2f}m) → 자동 Grasp")
                self._stop_nav()
                self._delivery_state = "ARRIVED"
                self._auto_state = "GRASPING"
                self._nav_wait_t = 0.0
            elif self._nav_wait_t > 60.0:
                # Nav2 없이도 대비: 직접 이동
                self._navigate_toward(*self.EXTINGUISHER_POS)

        elif self._auto_state == "GRASPING":
            # Grasp 상태머신 완료 대기 (이동 없음)
            if self._delivery_state == "SEARCHING" and self._has_object:
                print(f"\n[{self.namespace}] 파지 완료 → 테이블 탈출(후진)\n")
                self._backup_t = 0.0
                self._auto_state = "BACKUP"

        elif self._auto_state == "BACKUP":
            # 테이블 충돌 방지: 후진 2.5초 → Nav2 goal 전송
            self._backup_t += step_size
            if self._backup_t < 2.5:
                self._nav_command[0] = -0.4
                self._nav_command[1] = 0.0
                self._nav_command[2] = 0.0
            else:
                self._stop_nav()
                print(f"\n[{self.namespace}] 테이블 탈출 완료 → 화재 위치로 Nav2 이동\n")
                self._send_nav2_goal(*self.FIRE_POS)
                self._auto_state = "NAV_TO_FIRE"
                self._nav_wait_t = 0.0
                self._nav_idle_t = 0.0

        elif self._auto_state == "NAV_TO_FIRE":
            self._nav_wait_t += step_size
            dist = self._dist_to(*self.FIRE_POS)
            if dist < 3.0:
                print(f"[{self.namespace}] 화재 위치 도착 ({dist:.2f}m) → 자동 투척")
                self._stop_nav()
                self._delivery_state = "DROP_SETTLE"
                self._grasp_t = 0.0
                self._auto_state = "THROWING"
                self._nav_wait_t = 0.0
            elif self._nav_wait_t > 120.0:
                print(f"[{self.namespace}] 화재 이동 타임아웃 ({dist:.2f}m) → 강제 투척")
                self._stop_nav()
                self._delivery_state = "DROP_SETTLE"
                self._grasp_t = 0.0
                self._auto_state = "THROWING"
                self._nav_wait_t = 0.0
            else:
                # Nav2 UDP 수신이 없으면 직접 이동으로 폴백 (매 스텝 갱신)
                if self._udp_received:
                    self._nav_idle_t = 0.0
                else:
                    self._nav_idle_t += step_size
                    if self._nav_idle_t > 1.0:
                        self._navigate_toward(*self.FIRE_POS)

        elif self._auto_state == "THROWING":
            if self._delivery_state == "SEARCHING":
                print(f"\n[{self.namespace}] 투척 완료! 시나리오 종료\n")
                self._auto_state = "DONE"

    # ------------------------------------------------------------------ #
    # robot2 순찰
    # ------------------------------------------------------------------ #
    def _run_patrol_step(self, step_size):
        if not self._patrol_active:
            return
        # YOLO가 사람 발견 중이면 순찰 일시 정지 (Nav2 goal은 유지)
        if getattr(self, "_yolo_state", "SEARCHING") != "SEARCHING":
            return

        self._patrol_wait_t += step_size
        tx, ty = self._patrol_waypoints[self._patrol_idx]
        dist = self._dist_to(tx, ty)

        if dist < 2.0 or self._patrol_wait_t > 90.0:
            reason = "도착" if dist < 2.0 else "타임아웃"
            print(f"[{self.namespace}] 웨이포인트 {self._patrol_idx} {reason} ({dist:.1f}m) → 다음")
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_waypoints)
            nx, ny = self._patrol_waypoints[self._patrol_idx]
            self._send_nav2_goal(nx, ny)
            self._patrol_wait_t = 0.0
            self._nav_idle_t = 0.0
        else:
            # Nav2 UDP 수신이 없으면 직접 이동으로 폴백 (매 스텝 갱신)
            if self._udp_received:
                self._nav_idle_t = 0.0
            else:
                self._nav_idle_t += step_size
                if self._nav_idle_t > 1.0:
                    self._navigate_toward(tx, ty, max_linear=0.4)

    # ------------------------------------------------------------------ #
    # 포즈 파일 퍼블리시
    # ------------------------------------------------------------------ #
    def _publish_pose_file(self):
        try:
            pos, q = self._spot.robot.get_world_pose()
            payload = {
                "namespace": self.namespace,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "orientation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            }
            tmp = f"{self._pose_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, self._pose_file)
        except Exception as e:
            print(f"[{self.namespace}] pose file 쓰기 실패: {e}")

    # ------------------------------------------------------------------ #
    # 키보드: G = Grasp 시작, Q = 투척
    # ------------------------------------------------------------------ #
    def _on_agent_keyboard(self, event, *args, **kwargs) -> bool:
        import carb
        if not self.allow_grasp_trigger:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "G":
                if self._delivery_state == "SEARCHING":
                    # 거리 체크
                    try:
                        from omni.isaac.core.prims import XFormPrim
                        robot_prim = XFormPrim(f"/World/{self.namespace}/body")
                        cube_prim  = XFormPrim("/World/Cube")
                        r_pos, _ = robot_prim.get_world_pose()
                        c_pos, _ = cube_prim.get_world_pose()
                        dist = np.linalg.norm(r_pos[:2] - c_pos[:2])
                        if dist > 0.8:
                            print(f"[{self.namespace}] 소화기가 너무 멉니다 ({dist:.2f}m). 더 가까이 이동하세요.")
                            return True
                    except Exception:
                        pass
                    print(f"[{self.namespace}] G키 → Grasp 시퀀스 시작")
                    self._delivery_state = "ARRIVED"

            elif event.input.name == "Q":
                if self._delivery_state == "SEARCHING" and self._has_object:
                    print(f"[{self.namespace}] Q키 → 소화기 투척")
                    self._delivery_state = "DROP_SETTLE"
                    self._grasp_t = 0.0
        return True

    # ------------------------------------------------------------------ #
    # 내부 헬퍼
    # ------------------------------------------------------------------ #
    def _dp(self):
        return np.array(self._spot.default_pos, dtype=np.float32).reshape(-1)

    def _hold(self, arm6, grip):
        from isaacsim.core.utils.types import ArticulationAction
        full = self._stance.copy()
        for k, idx in enumerate(self.ARM_IDX):
            full[idx] = arm6[k]
        full[self.GRIP_IDX] = grip
        self._spot.robot.apply_action(ArticulationAction(joint_positions=full))

    def _arm_override(self, arm6, grip):
        from isaacsim.core.utils.types import ArticulationAction
        vals = np.array(list(arm6) + [grip], dtype=np.float32)
        idxs = np.array(self.ARM_IDX + [self.GRIP_IDX])
        self._spot.robot.apply_action(
            ArticulationAction(joint_positions=vals, joint_indices=idxs)
        )

    def _set_heavy_mode(self, enable: bool):
        if getattr(self, "_is_heavy_mode", False) == enable:
            return
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        factor = 3.0 if enable else (1.0 / 3.0)
        leg_kw = ["hip", "uleg", "lleg"]
        for prim in stage.TraverseAll():
            path = prim.GetPath().pathString
            n = prim.GetName()
            if not path.startswith(f"/World/{self.namespace}"):
                continue
            if any(k in n for k in leg_kw) or n == "body":
                mp = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if mp:
                    cur = mp.GetMassAttr().Get()
                    if cur is not None:
                        mp.GetMassAttr().Set(float(cur * factor))
        self._is_heavy_mode = enable
        print(f"[{self.namespace}] 질량 {'x3.0 적용' if enable else '원상복구'}")

    # ------------------------------------------------------------------ #
    # 센서 셋업
    # ------------------------------------------------------------------ #
    def setup_sensors(self):
        stage = omni.usd.get_context().get_stage()
        lidar_parent = f"/World/{self.namespace}/body"

        success, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="Functional_Lidar",
            parent=lidar_parent,
            min_range=0.65,
            max_range=20.0,
            draw_points=False,
            draw_lines=False,
            horizontal_fov=360.0,
            vertical_fov=1.0,
            horizontal_resolution=0.4,
            vertical_resolution=1.0,
            rotation_rate=0.0,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False,
        )

        if success:
            sensor_prim_path = lidar.GetPath()
            lidar.GetPrim().GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(0.0, 0.0, 0.25)
            )
            print(f"[{self.namespace}] Lidar 생성: {sensor_prim_path}")
        else:
            print(f"[{self.namespace}] Lidar 생성 실패")
            sensor_prim_path = None

        self._setup_ros2_graph(sensor_prim_path)

        # YOLO 그리퍼 카메라 (robot2 전용)
        if not self.allow_grasp_trigger:
            self._setup_yolo_camera()

        # VisualGraspCube (robot1 전용)
        if self.allow_grasp_trigger:
            try:
                cube_path = "/World/VisualGraspCube"
                from omni.isaac.core.utils.stage import add_reference_to_stage
                from pxr import UsdGeom
                import os as _os
                base_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                ext_usd = _os.path.join(
                    base_dir, "map", "fire_extinguisher", "World0.usd"
                )
                if not stage.GetPrimAtPath(cube_path).IsValid():
                    add_reference_to_stage(ext_usd, cube_path)
                v_prim = stage.GetPrimAtPath(cube_path)
                if v_prim.IsValid():
                    UsdGeom.Imageable(v_prim).MakeInvisible()
                    xform = UsdGeom.Xformable(v_prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -10.0))
                self._visual_cube_path = cube_path
                print(f"[{self.namespace}] VisualGraspCube 생성 완료: {cube_path}")
            except Exception as e:
                print(f"[{self.namespace}] VisualGraspCube 생성 실패: {e}")

        # Headlight
        light_path = f"{lidar_parent}/Headlight"
        headlight = UsdLux.SphereLight.Define(stage, light_path)
        headlight.CreateIntensityAttr(250000)
        headlight.GetPrim().CreateAttribute(
            "exposure", Sdf.ValueTypeNames.Float
        ).Set(5.0)
        headlight.CreateRadiusAttr(0.05)
        headlight.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.8))
        headlight.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.1))
        print(f"[{self.namespace}] 센서 셋업 완료")

    def _setup_yolo_camera(self):
        self._camera_gripper = None
        self._camera_initialized = False
        self._yolo_counter = 0
        self._yolo_state = "SEARCHING"
        self._tracking_command = np.zeros(3)
        self._was_person_detected = False

        try:
            from omni.isaac.sensor import Camera
            cam_path = f"/World/{self.namespace}/arm0_link_wr1/gripper_camera"
            self._camera_gripper = Camera(
                prim_path=cam_path,
                resolution=(320, 240),
                translation=np.array([0.1, 0.0, 0.0]),
            )
            # Isaac Sim 두 번째 뷰포트에 카메라 연결
            try:
                import omni.kit.viewport.utility as vp_util
                vp = vp_util.create_viewport_window("robot2 Gripper View", width=320, height=240)
                if vp:
                    vp.viewport_api.set_active_camera(cam_path)
                    print(f"[{self.namespace}] 뷰포트 창 생성: robot2 Gripper View")
            except Exception as ve:
                print(f"[{self.namespace}] 뷰포트 창 생성 실패 (무시): {ve}")
        except Exception as e:
            print(f"[{self.namespace}] 그리퍼 카메라 생성 실패: {e}")

        self.yolo_enabled = False
        try:
            from ultralytics import YOLO
            import os as _os
            model_path = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "yolov8n.pt",
            )
            self.yolo_model = YOLO(model_path)
            self.yolo_enabled = True
            print(f"[{self.namespace}] YOLOv8n 로드 완료: {model_path}")
        except Exception as e:
            print(f"[{self.namespace}] YOLO 로드 실패: {e}")

    def _setup_ros2_graph(self, sensor_prim_path):
        import omni.graph.core as og
        from omni.isaac.core.utils.extensions import enable_extension

        enable_extension("isaacsim.core.nodes")
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.physx")

        try:
            keys = og.Controller.Keys
            stage = omni.usd.get_context().get_stage()

            nodes = [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("ReadLidar", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                ("PublishScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
            ]

            conns = [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                ("OnPlaybackTick.outputs:tick", "ReadLidar.inputs:execIn"),
                ("ReadLidar.outputs:execOut", "PublishScan.inputs:execIn"),
                ("Context.outputs:context", "PublishScan.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishScan.inputs:timeStamp"),
                ("ReadLidar.outputs:azimuthRange", "PublishScan.inputs:azimuthRange"),
                ("ReadLidar.outputs:depthRange", "PublishScan.inputs:depthRange"),
                ("ReadLidar.outputs:horizontalFov", "PublishScan.inputs:horizontalFov"),
                ("ReadLidar.outputs:horizontalResolution", "PublishScan.inputs:horizontalResolution"),
                ("ReadLidar.outputs:intensitiesData", "PublishScan.inputs:intensitiesData"),
                ("ReadLidar.outputs:linearDepthData", "PublishScan.inputs:linearDepthData"),
                ("ReadLidar.outputs:numCols", "PublishScan.inputs:numCols"),
                ("ReadLidar.outputs:numRows", "PublishScan.inputs:numRows"),
                ("ReadLidar.outputs:rotationRate", "PublishScan.inputs:rotationRate"),
            ]

            vals = [
                ("SubscribeTwist.inputs:topicName", f"{self.namespace}/cmd_vel"),
                ("PublishScan.inputs:topicName", f"{self.namespace}/scan"),
                ("PublishScan.inputs:frameId", f"{self.namespace}/Functional_Lidar"),
            ]

            if sensor_prim_path:
                import usdrt.Sdf
                path_str = (
                    str(sensor_prim_path.GetPath())
                    if hasattr(sensor_prim_path, "GetPath")
                    else str(sensor_prim_path)
                )
                vals.append(("ReadLidar.inputs:lidarPrim", [usdrt.Sdf.Path(path_str)]))

            og.Controller.edit(
                {
                    "graph_path": f"/ROS2_Graph_{self.namespace}",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: nodes,
                    keys.CONNECT: conns,
                    keys.SET_VALUES: vals,
                },
            )
            print(f"[{self.namespace}] ROS2 Graph 생성 완료")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{self.namespace}] ROS2 Graph 생성 실패: {e}")

    # ------------------------------------------------------------------ #
    # 물리 스텝
    # ------------------------------------------------------------------ #
    def on_physics_step(self, step_size, base_command) -> None:
        import struct

        self._publish_pose_file()

        # 투척 후 소화기 회전 이펙트
        if self._is_thrown and hasattr(self, "_thrown_cube"):
            try:
                vel = self._thrown_cube.get_angular_velocity()
                if vel is not None:
                    vel[0] = vel[0] * 0.95
                    vel[1] = vel[1] * 0.95
                    vel[2] = 25.0
                    self._thrown_cube.set_angular_velocity(vel)
            except Exception:
                pass

        # UDP cmd_vel 수신
        self._udp_received = False  # 이 스텝에서 Nav2 명령 수신 여부
        try:
            while True:
                data, _ = self.udp_sock.recvfrom(1024)
                if len(data) >= 12:
                    vx, vy, wz = struct.unpack("fff", data[:12])
                    # RL 정책 안정성: wz 과도 시 gait 붕괴 방지 (키보드 0.4 기준)
                    wz = float(np.clip(wz, -0.5, 0.5))
                    vx = float(np.clip(vx, -0.8, 0.8))
                    vy = float(np.clip(vy, -0.4, 0.4))
                    self._nav_command[0] = vx
                    self._nav_command[1] = vy
                    self._nav_command[2] = wz
                    self._udp_received = True
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[{self.namespace}] UDP 읽기 오류: {e}")

        # 첫 스텝 초기화
        if self.first_step:
            self._spot.initialize()
            self._spot.robot.set_joint_positions(self._spot.default_pos)
            self._spot.robot.set_joint_velocities(self._spot.default_vel)
            self.first_step = False
            print(f"[{self.namespace}] 로봇 초기화 완료")

            # initialpose → Nav2
            try:
                import subprocess
                pos, q = self._spot.robot.get_world_pose()
                x, y = pos[0], pos[1]
                msg = (
                    f"{{header: {{frame_id: 'map'}}, pose: {{pose: {{position: "
                    f"{{x: {x}, y: {y}, z: 0.0}}, orientation: {{w: {q[0]}, "
                    f"x: {q[1]}, y: {q[2]}, z: {q[3]}}}}}, covariance: "
                    f"[0.25,0,0,0,0,0,0,0.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
                    f"0,0,0,0,0,0,0,0,0,0,0,0.068]}}}}"
                )
                env = os.environ.copy()
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = ":".join(
                        p for p in env["PYTHONPATH"].split(":")
                        if "isaacsim" not in p.lower()
                    )
                ns = self.namespace
                subprocess.Popen(
                    ["timeout", "2", "ros2", "topic", "pub", "--once",
                     f"/{ns}/initialpose",
                     "geometry_msgs/msg/PoseWithCovarianceStamped", msg],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[{self.namespace}] initialpose 전송: x={x:.2f}, y={y:.2f}")
            except Exception as e:
                print(f"[{self.namespace}] initialpose 전송 실패: {e}")

            # Grasp용 게인 초기화 (robot1 전용)
            if self.allow_grasp_trigger:
                try:
                    dp = self._dp()
                    names = (
                        list(self._spot.robot.dof_names)
                        if hasattr(self._spot.robot, "dof_names")
                        else self._spot.robot._articulation_view.joint_names
                    )
                    stance = dp.copy()
                    for i, nm in enumerate(names):
                        if "hx" in nm:
                            stance[i] = 0.0
                        elif "hy" in nm:
                            stance[i] = 0.8
                        elif "kn" in nm:
                            stance[i] = -1.5
                    self._stance = stance
                    self._cur_arm = stance[self.ARM_IDX].copy()
                    self._cur_grip = -1.571
                    self._kps0, self._kds0 = (
                        self._spot.robot._articulation_view.get_gains()
                    )
                    print(f"[{self.namespace}] Grasp 초기화 완료 — G키: Grasp, Q키: 투척")
                except Exception as e:
                    print(f"[{self.namespace}] Grasp 초기화 오류: {e}")
            else:
                print(f"[{self.namespace}] 초기화 완료 — 화재 신호 대기 중")
            return

        # robot2: 순찰 + YOLO 상태머신
        if not self.allow_grasp_trigger:
            self._run_patrol_step(step_size)
            # 순찰이 _nav_command 갱신 후 total_cmd 재계산
            total_cmd = base_command + self._nav_command
            self._run_yolo_step(step_size, total_cmd)
            return

        # robot1: 자동 시나리오 (_nav_command 갱신)
        self._run_auto_scenario(step_size)
        # 자동 시나리오가 _nav_command 갱신 후 total_cmd 재계산
        total_cmd = base_command + self._nav_command

        # -------- robot1 Grasp 상태머신 --------
        if self._delivery_state == "SEARCHING":
            self._spot.forward(step_size, total_cmd)
            if self._carry_arm is not None:
                self._carry_arm += np.clip(
                    self._carry_tgt - self._carry_arm, -0.01, 0.01
                )
                self._arm_override(self._carry_arm, self.GRIP_CLOSE)
            elif self._has_object:
                self._spot.override_arm_angles = None
                self._spot.override_grip_angle = self.GRIP_CLOSE

        elif self._delivery_state == "ARRIVED":
            print(f"\n[{self.namespace}] 목적지 도착 → SETTLE\n")
            self._delivery_state = "SETTLE"
            self._grasp_t = 0.0
            try:
                nd = len(self._dp())
                kps = np.full(nd, 2000.0)
                kds = np.full(nd, 100.0)
                for idx in self.ARM_IDX:
                    kps[idx] = 5000.0
                    kds[idx] = 250.0
                kps[self.GRIP_IDX] = 5000.0
                kds[self.GRIP_IDX] = 250.0
                self._spot.robot._articulation_view.set_gains(
                    kps=kps.reshape(1, -1), kds=kds.reshape(1, -1)
                )
                self._cur_arm = self._stance[self.ARM_IDX].copy()
                self._cur_grip = -1.571
            except Exception as e:
                print(f"[{self.namespace}] 강성 부스트 오류: {e}")

        elif self._delivery_state == "SETTLE":
            self._hold(self._cur_arm, self._cur_grip)
            self._grasp_t += step_size
            if self._grasp_t > 2.5:
                self._delivery_state = "HOVER"
                self._grasp_t = 0.0
                print(f"[{self.namespace}] SETTLE 완료 → HOVER")

        elif self._delivery_state == "DROP_SETTLE":
            try:
                nd = len(self._dp())
                kps = np.full(nd, 2000.0)
                kds = np.full(nd, 100.0)
                for idx in self.ARM_IDX:
                    kps[idx] = 5000.0
                    kds[idx] = 250.0
                kps[self.GRIP_IDX] = 5000.0
                kds[self.GRIP_IDX] = 250.0
                self._spot.robot._articulation_view.set_gains(
                    kps=kps.reshape(1, -1), kds=kds.reshape(1, -1)
                )
                current_positions = self._spot.robot.get_joint_positions()
                self._cur_arm = current_positions[self.ARM_IDX].copy()
                self._cur_grip = self.GRIP_CLOSE
                print(f"\n[{self.namespace}] DROP_SETTLE → DROP_REACH\n")
            except Exception:
                pass
            self._delivery_state = "DROP_REACH"
            self._grasp_t = 0.0

        elif self._delivery_state in [
            "HOVER", "GRASP", "CLOSE", "LIFT", "DONE",
            "DROP_REACH", "DROP_OPEN", "DROP_DONE",
        ]:
            self._grasp_t += step_size

            POSES = [
                (np.array([0.0, -0.85, 1.41, 0.0, 1.05, 0.0], dtype=np.float32), -1.571),  # 0: Hover
                (np.array([0.0, -0.65, 1.41, 0.0, 0.95, 0.0], dtype=np.float32), -1.571),  # 1: Grasp
                (np.array([0.0, -2.95, 2.95, 0.0, 1.20, 0.0], dtype=np.float32), -0.5),    # 2: Lift
            ]

            targets = {
                "HOVER": (POSES[0][0], -1.571, "GRASP"),
                "GRASP": (POSES[1][0], -1.571, "CLOSE"),
                "LIFT":  (POSES[2][0], self.GRIP_CLOSE, "DONE"),
            }

            if self._delivery_state in targets:
                tgt_arm, grip, nxt = targets[self._delivery_state]
                self._cur_arm += np.clip(tgt_arm - self._cur_arm, -0.02, 0.02)
                self._hold(self._cur_arm, grip)
                if (
                    np.max(np.abs(tgt_arm - self._cur_arm)) < 0.02
                    and self._grasp_t > 1.0
                ):
                    print(f"[{self.namespace}] {self._delivery_state} → {nxt}")
                    self._delivery_state = nxt
                    self._grasp_t = 0.0

            elif self._delivery_state == "CLOSE":
                self._cur_grip = max(self.GRIP_CLOSE, self._cur_grip - 0.03)
                self._hold(POSES[1][0], self._cur_grip)
                if self._cur_grip <= self.GRIP_CLOSE + 1e-3 and self._grasp_t > 1.5:
                    self._cur_arm = POSES[1][0].copy()
                    print(f"[{self.namespace}] CLOSE 완료 — Magnetic Grasp 활성화")
                    # Magnetic Grasp: 물리 큐브 Z=-10 숨기고 시각 큐브 표시
                    self._grabbed_cube_path = getattr(
                        self, "_visual_cube_path", "/World/VisualGraspCube"
                    )
                    try:
                        from omni.isaac.core.prims import XFormPrim
                        from pxr import UsdGeom
                        stage = omni.usd.get_context().get_stage()

                        real_cube = XFormPrim("/World/Cube")
                        real_cube.set_world_pose(position=np.array([0.0, 0.0, -10.0]))

                        v_prim = stage.GetPrimAtPath(self._grabbed_cube_path)
                        if v_prim.IsValid():
                            UsdGeom.Imageable(v_prim).MakeVisible()
                        print(f"[{self.namespace}] 소화기 집기 완료!")
                    except Exception as e:
                        print(f"[{self.namespace}] Grasp 스왑 실패: {e}")
                    self._delivery_state = "LIFT"
                    self._grasp_t = 0.0

            elif self._delivery_state == "DONE":
                self._hold(POSES[2][0], self.GRIP_CLOSE)
                if self._grasp_t > 1.5:
                    print(f"[{self.namespace}] DONE → FOLD_ARM")
                    self._delivery_state = "FOLD_ARM"
                    self._grasp_t = 0.0
                    self._carry_arm = POSES[2][0].copy()
                    self._carry_tgt = self._dp()[self.ARM_IDX].copy()

            elif self._delivery_state == "DROP_REACH":
                tgt_arm = POSES[1][0]
                self._cur_arm += np.clip(tgt_arm - self._cur_arm, -0.02, 0.02)
                self._hold(self._cur_arm, self.GRIP_CLOSE)
                if (
                    np.max(np.abs(tgt_arm - self._cur_arm)) < 0.02
                    and self._grasp_t > 1.5
                ):
                    print(f"[{self.namespace}] DROP_REACH → DROP_OPEN")
                    self._delivery_state = "DROP_OPEN"
                    self._grasp_t = 0.0

            elif self._delivery_state == "DROP_OPEN":
                self._cur_grip = max(-1.571, self._cur_grip - 0.03)
                self._hold(self._cur_arm, self._cur_grip)
                if self._cur_grip <= -1.571 + 1e-3 and self._grasp_t > 1.0:
                    print(f"[{self.namespace}] DROP_OPEN → 투척!")
                    if self._grabbed_cube_path is not None:
                        try:
                            from omni.isaac.core.prims import XFormPrim
                            from omni.isaac.core.prims.rigid_prim import RigidPrim
                            from pxr import UsdGeom
                            stage = omni.usd.get_context().get_stage()

                            vis_cube = XFormPrim(self._grabbed_cube_path)
                            pos, rot = vis_cube.get_world_pose()

                            real_cube = RigidPrim("/World/Cube")
                            real_cube.initialize()

                            body_prim = XFormPrim(f"/World/{self.namespace}/body")
                            body_pos, _ = body_prim.get_world_pose()

                            dir_vec = pos[:2] - body_pos[:2]
                            dist = np.linalg.norm(dir_vec)

                            # 충돌 방지: 던지는 방향으로 25cm 오프셋
                            safe_pos = pos.copy()
                            if dist > 0.01:
                                safe_pos[0] += (dir_vec[0] / dist) * 0.25
                                safe_pos[1] += (dir_vec[1] / dist) * 0.25

                            real_cube.set_world_pose(position=safe_pos, orientation=rot)

                            self._is_thrown = True
                            self._thrown_cube = real_cube

                            if dist > 0.01:
                                dir_vec = dir_vec / dist
                            else:
                                dir_vec = np.array([1.0, 0.0])

                            throw_vel = np.array([
                                dir_vec[0] * 2.5,
                                dir_vec[1] * 2.5,
                                1.0,
                            ])
                            real_cube.set_linear_velocity(throw_vel)

                            v_prim = stage.GetPrimAtPath(self._grabbed_cube_path)
                            if v_prim.IsValid():
                                UsdGeom.Imageable(v_prim).MakeInvisible()
                            print(f"[{self.namespace}] 소화기 투척 완료!")
                        except Exception as e:
                            print(f"[{self.namespace}] 투척 실패: {e}")
                        self._grabbed_cube_path = None
                    self._delivery_state = "DROP_DONE"
                    self._grasp_t = 0.0

            elif self._delivery_state == "DROP_DONE":
                tgt_arm = POSES[2][0]
                self._cur_arm += np.clip(tgt_arm - self._cur_arm, -0.02, 0.02)
                self._hold(self._cur_arm, -1.571)
                if (
                    np.max(np.abs(tgt_arm - self._cur_arm)) < 0.02
                    and self._grasp_t > 1.5
                ):
                    print(f"[{self.namespace}] DROP_DONE → FOLD_ARM")
                    self._delivery_state = "FOLD_ARM"
                    self._grasp_t = 0.0
                    self._carry_arm = POSES[2][0].copy()
                    self._carry_tgt = self._dp()[self.ARM_IDX].copy()

        elif self._delivery_state == "FOLD_ARM":
            self._carry_arm += np.clip(
                self._carry_tgt - self._carry_arm, -0.01, 0.01
            )
            self._hold(self._carry_arm, self.GRIP_CLOSE)
            if np.max(np.abs(self._carry_tgt - self._carry_arm)) < 0.05:
                print(f"\n[{self.namespace}] 팔 접기 완료 → SEARCHING\n")
                try:
                    kps = np.array(self._kps0, dtype=np.float32).reshape(1, -1)
                    kds = np.array(self._kds0, dtype=np.float32).reshape(1, -1)
                    kps[0, self.GRIP_IDX] = 5000.0
                    kds[0, self.GRIP_IDX] = 250.0
                    self._spot.robot._articulation_view.set_gains(kps=kps, kds=kds)
                except Exception:
                    pass
                self._set_heavy_mode(False)
                self._carry_arm = None
                self._has_object = self._grabbed_cube_path is not None
                if hasattr(self._spot, "action") and hasattr(self._spot, "_action_scale"):
                    current_positions = self._spot.robot.get_joint_positions()
                    self._spot.action = (
                        (current_positions - self._spot.default_pos)
                        / self._spot._action_scale
                    )
                    self._spot._previous_action = self._spot.action.copy()
                self._delivery_state = "SEARCHING"

    def _run_yolo_step(self, step_size, base_command):
        """robot2 전용: YOLO 인명탐지 상태머신 + forward"""

        if not self.yolo_enabled or not hasattr(self, "_camera_gripper") or self._camera_gripper is None:
            self._spot.forward(step_size, base_command)
            return

        # 카메라 초기화 (첫 호출 시)
        if not self._camera_initialized:
            try:
                self._camera_gripper.initialize()
                try:
                    self._camera_gripper.add_distance_to_image_plane_to_frame()
                except AttributeError:
                    try:
                        self._camera_gripper.add_distance_to_camera_to_frame()
                    except Exception:
                        pass
                self._camera_initialized = True
                print(f"[{self.namespace}] 그리퍼 카메라 초기화 완료")
            except Exception as e:
                print(f"[{self.namespace}] 카메라 초기화 실패: {e}")
                self._spot.forward(step_size, base_command)
                return

        self._yolo_counter += 1

        # ~4Hz 주기로 비전 처리
        if self._yolo_counter % 15 == 0:
            try:
                img   = self._camera_gripper.get_rgba()
                depth = self._camera_gripper.get_depth()

                found, best_depth, best_cx, img_plot = False, 999.0, 160, None

                if img is not None and depth is not None:
                    # cv2.imshow는 Isaac Sim 내부 Qt와 충돌하므로 사용 금지
                    # numpy로 직접 BGR 변환 없이 RGB 그대로 YOLO에 전달
                    img_rgb = img[:, :, :3]
                    results = self.yolo_model.predict(source=img_rgb, conf=0.45, verbose=False)
                    max_area = 0
                    for r in results:
                        if r.boxes is not None:
                            for i, c in enumerate(r.boxes.cls):
                                if int(c) == 0:  # person
                                    box = r.boxes.xyxy[i].cpu().numpy()
                                    area = (box[2] - box[0]) * (box[3] - box[1])
                                    if area > max_area:
                                        max_area = area
                                        cx = int((box[0] + box[2]) / 2.0)
                                        cy = int((box[1] + box[3]) / 2.0)
                                        cy = int(np.clip(cy, 0, depth.shape[0] - 1))
                                        cx = int(np.clip(cx, 0, depth.shape[1] - 1))
                                        d = float(depth[cy, cx])
                                        if 0.01 < d < 15.0:
                                            best_depth = d
                                            best_cx    = cx
                                            found      = True

                if found:
                    print(f"[{self.namespace}] 사람 감지: {best_depth:.2f}m cx={best_cx} [{self._yolo_state}]")

                # ---------- 상태머신 ----------
                if found:
                    self._was_person_detected = True
                    self._person_focus_end = getattr(self, "_sim_time_r2", 0.0) + 5.0
                else:
                    if self._was_person_detected:
                        print(f"[{self.namespace}] 사람 시야 이탈 — 추적 유지")
                        self._was_person_detected = False

                if self._yolo_state == "SEARCHING":
                    if found:
                        self._yolo_state = "ALIGNING"
                        print(f"\n[{self.namespace}] 👤 사람 발견! 정렬 시작\n")
                    else:
                        self._tracking_command = np.zeros(3)

                elif self._yolo_state == "ALIGNING":
                    if found:
                        target_turn = (160 - best_cx) / 160.0
                        turn_speed  = np.clip(target_turn * 0.8, -0.4, 0.4)
                        if abs(target_turn) >= 0.08 and abs(turn_speed) < 0.15:
                            turn_speed = 0.15 if turn_speed > 0 else -0.15
                        self._tracking_command = np.array([0.0, 0.0, turn_speed])

                        tol = 0.15 if getattr(self, "_centered_locked", False) else 0.08
                        if abs(target_turn) < tol:
                            if not getattr(self, "_centered_locked", False):
                                travel = max(0.0, best_depth - 2.0)
                                self._approach_start  = getattr(self, "_sim_time_r2", 0.0)
                                self._approach_dur    = travel / 0.5
                                self._centered_locked = True
                                self._yolo_state      = "APPROACHING"
                                print(f"[{self.namespace}] 🎯 정렬 완료! {travel:.2f}m 전진")
                        else:
                            self._centered_locked = False
                    else:
                        t_now = getattr(self, "_sim_time_r2", 0.0)
                        if getattr(self, "_person_focus_end", 0.0) <= t_now:
                            self._yolo_state = "SEARCHING"

                elif self._yolo_state == "APPROACHING":
                    t_now    = getattr(self, "_sim_time_r2", 0.0)
                    elapsed  = t_now - getattr(self, "_approach_start", t_now)
                    if elapsed >= getattr(self, "_approach_dur", 0.0):
                        print(f"\n[{self.namespace}] 🎉 목적지 도착! 대기\n")
                        self._yolo_state       = "ARRIVED"
                        self._tracking_command = np.zeros(3)
                    else:
                        self._tracking_command = np.array([0.5, 0.0, 0.0])

                elif self._yolo_state == "ARRIVED":
                    self._tracking_command = np.zeros(3)

            except Exception as e:
                print(f"[{self.namespace}] YOLO 처리 오류: {e}")

        # 시뮬레이션 시간 누적 (상태머신 타이머용)
        self._sim_time_r2 = getattr(self, "_sim_time_r2", 0.0) + step_size

        # 키보드 입력 시 수동 조종으로 복귀
        if np.any(base_command != 0) and self._yolo_state != "SEARCHING":
            print(f"[{self.namespace}] 키보드 감지 → 수동 조종")
            self._yolo_state = "SEARCHING"

        # 자율 추적 명령 우선 적용
        if self._yolo_state != "SEARCHING":
            cmd = self._tracking_command.copy()
        else:
            cmd = base_command if np.any(base_command != 0) else self._tracking_command.copy()

        self._spot.forward(step_size, cmd)

        # Magnetic Grasp Follow: 시각 큐브를 그리퍼에 붙임
        if self._grabbed_cube_path is not None:
            try:
                from omni.isaac.core.prims import XFormPrim
                gripper_prim = XFormPrim(f"/World/{self.namespace}/arm0_link_wr1")
                cube_prim    = XFormPrim(self._grabbed_cube_path)
                pos, rot = gripper_prim.get_world_pose()
                q_vec = rot[1:]
                q_w   = rot[0]
                v = np.array([0.15, 0.0, -0.15])
                t = 2.0 * np.cross(q_vec, v)
                offset = v + q_w * t + np.cross(q_vec, t)
                cube_prim.set_world_pose(position=pos + offset, orientation=rot)
            except Exception:
                pass
