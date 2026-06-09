import tkinter as tk
from tkinter import ttk
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import threading

class DashboardNode(Node):
    def __init__(self, ui):
        super().__init__('main_dashboard_node')
        self.ui = ui
        # 화재 알람 퍼블리셔
        self.fire_pub = self.create_publisher(String, '/fire_alarm', 10)
        # 로봇 상태 구독
        self.arm_sub = self.create_subscription(String, '/spot1/arm_status', self.arm_callback, 10)
        self.yolo_sub = self.create_subscription(String, '/spot2/yolo/detected_person', self.yolo_callback, 10)

    def trigger_fire(self, room="room_a"):
        msg = String()
        msg.data = room
        self.fire_pub.publish(msg)
        self.get_logger().info(f"🔥 Fire alarm triggered: {room}")

    def arm_callback(self, msg):
        status = msg.data
        if "GRASPING" in status:
            self.ui.update_spot1_status("📦 소화기 파지 완료! 화재 구역 이동 중")
        elif "PLACING" in status:
            self.ui.update_spot1_status("✅ 진압 완료 (소화기 배치)")

    def yolo_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("detected"):
                self.ui.update_spot2_status("⚠️ 조난자 발견! 동적 접근 및 유도 중...")
            else:
                if "조난자" not in self.ui.lbl_spot2.cget("text"):
                    self.ui.update_spot2_status("🔍 각 방 순회 및 인명 탐색 중...")
        except Exception as e:
            pass

class DashboardApp:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("🔥 TEAM 핫스팟 - 디지털 트윈 화재 대응 관제탑")
        self.root.geometry("650x400")
        self.root.configure(bg="#1e293b") # 다크 테마 배경

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#1e293b", foreground="#f8fafc", font=("Noto Sans KR", 12))
        style.configure("Header.TLabel", font=("Noto Sans KR", 20, "bold"), foreground="#38bdf8")

        # 헤더
        ttk.Label(root, text="디지털 트윈 화재 관제 시스템", style="Header.TLabel").pack(pady=25)

        # 화재 트리거 버튼
        self.fire_btn = tk.Button(root, text="🚨 화재 발생 트리거 (Room A)", font=("Noto Sans KR", 16, "bold"),
                                  bg="#ef4444", fg="white", activebackground="#dc2626", borderwidth=0,
                                  command=self.trigger_alarm, cursor="hand2", pady=10)
        self.fire_btn.pack(fill=tk.X, padx=50, pady=10)

        # 상태 모니터링 프레임
        status_frame = tk.Frame(root, bg="#334155", bd=0)
        status_frame.pack(pady=25, fill=tk.BOTH, expand=True, padx=50)

        # Spot 1 상태
        ttk.Label(status_frame, text="🦾 [Spot 1] 진압조 상태 :", background="#334155", font=("Noto Sans KR", 12, "bold")).grid(row=0, column=0, sticky="w", padx=20, pady=20)
        self.lbl_spot1 = ttk.Label(status_frame, text="🟢 대기 중 (IDLE)", background="#334155", foreground="#94a3b8")
        self.lbl_spot1.grid(row=0, column=1, sticky="w", padx=10, pady=20)

        # Spot 2 상태
        ttk.Label(status_frame, text="👁️ [Spot 2] 구조조 상태 :", background="#334155", font=("Noto Sans KR", 12, "bold")).grid(row=1, column=0, sticky="w", padx=20, pady=20)
        self.lbl_spot2 = ttk.Label(status_frame, text="🟢 대기 중 (IDLE)", background="#334155", foreground="#94a3b8")
        self.lbl_spot2.grid(row=1, column=1, sticky="w", padx=10, pady=20)

    def trigger_alarm(self):
        self.lbl_spot1.config(text="🚨 소화기 위치로 출동 중...", foreground="#fbbf24")
        self.lbl_spot2.config(text="🚨 순찰 구역으로 출동 중...", foreground="#fbbf24")
        self.node.trigger_fire("room_a")
        
        # 버튼 비활성화 (중복 클릭 방지)
        self.fire_btn.config(state=tk.DISABLED, bg="#7f1d1d", text="🔥 재난 대응 시나리오 가동 중...")

    def update_spot1_status(self, text):
        self.lbl_spot1.config(text=text, foreground="#4ade80")

    def update_spot2_status(self, text):
        if "발견" in text:
            self.lbl_spot2.config(text=text, foreground="#f87171")
        else:
            self.lbl_spot2.config(text=text, foreground="#4ade80")

def ros_spin_thread(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    root = tk.Tk()
    node = DashboardNode(None)
    app = DashboardApp(root, node)
    node.ui = app

    # ROS2 노드는 별도의 스레드에서 실행 (GUI 멈춤 방지)
    thread = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    thread.start()

    # GUI 메인 루프 시작
    root.mainloop()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
