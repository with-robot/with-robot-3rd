# Select the goals for path planning
# Conducting path planning for just one goal.
# path planning mode 를 두 가지로 설정할 수 있도록 함.. : (1) loop waypoints, (2) one goal
# path planning 실패 시 위치 조정해서 재시도하는 기능 추가해야함.

import threading
import time
import logging
from youBot import YouBot
from grid_mapping import MappingBot
from pathplanning_multimode import youBotPP
from mcl_coppelia import LocalizationBot


class SLAMController:
    def __init__(self):
        self.planning = youBotPP()
        self.mapping = MappingBot()
        self.localization = LocalizationBot()

        self.stop_event = threading.Event()  # 종료 이벤트
        self.mapping_mode = threading.Event()  # 맵핑 모드
        self.localization_mode = threading.Event()  # 로컬라이제이션 모드

        self.target_points = []  # 목표 지점 리스트
        self.predefined_points = {
            "bedroom1": (1.0, 2.0),
            "bedroom2": (2.0, 3.0),
            "toilet": (1.5, 1.0),
            "entrance": (0.0, 0.0),
            "dining": (2.5, 2.5),
            "livingroom": (3.0, 2.0),
            "balcony_init": (3.5, 1.5),
            "balcony_end": (4.0, 1.0),
        }

    def init_coppelia(self):
        """CoppeliaSim 초기화"""
        logging.info("Initializing CoppeliaSim...")
        self.planning.init_coppelia()
        self.mapping.init_coppelia()
        self.localization.init_coppelia()

    def run_planning(self):
        """경로 계획 실행"""
        logging.info("Starting Path Planning...")
        while not self.stop_event.is_set():
            if self.mapping_mode.is_set():
                # Waypoints 순찰
                self.planning.set_mode("mapping")
                self.planning.run_coppelia()
            elif self.localization_mode.is_set() and self.target_points:
                # Single goal path planning
                goal = self.target_points[0]  # Use the first target point
                self.planning.set_mode("localization", goal)
                self.planning.run_coppelia()
            time.sleep(0.1)

    def run_mapping(self):
        """맵핑 실행"""
        logging.info("Starting Mapping...")
        while not self.stop_event.is_set():
            if self.mapping_mode.is_set():  # Mapping 모드 활성화 상태 확인
                self.mapping.run_coppelia()
            time.sleep(0.1)

    def run_localization(self):
        """로컬라이제이션 실행"""
        logging.info("Starting Localization...")
        while not self.stop_event.is_set():
            if self.localization_mode.is_set():  # Localization 모드 활성화 상태 확인
                self.localization.run_step()
            time.sleep(0.1)

    def start_threads(self):
        """모든 작업 쓰레드 시작"""
        self.init_thread = threading.Thread(target=self.init_coppelia, daemon=True)
        self.planning_thread = threading.Thread(target=self.run_planning, daemon=True)
        self.mapping_thread = threading.Thread(target=self.run_mapping, daemon=True)
        self.localization_thread = threading.Thread(target=self.run_localization, daemon=True)

        # 초기화 쓰레드는 join을 호출하여 동기화
        self.init_thread.start()
        self.init_thread.join()

        # 실행 쓰레드 시작
        self.planning_thread.start()
        self.mapping_thread.start()
        self.localization_thread.start()

    def stop_threads(self):
        """모든 작업 쓰레드 종료"""
        self.stop_event.set()
        self.planning_thread.join()
        self.mapping_thread.join()
        self.localization_thread.join()
        logging.info("All threads stopped.")

    def enable_mapping(self):
        """Mapping 모드 활성화"""
        logging.info("Mapping mode enabled.")
        self.mapping_mode.set()

    def disable_mapping(self):
        """Mapping 모드 비활성화"""
        logging.info("Mapping mode disabled.")
        self.mapping_mode.clear()

    def enable_localization(self):
        """Localization 모드 활성화"""
        logging.info("Localization mode enabled.")
        self.localization_mode.set()

    def disable_localization(self):
        """Localization 모드 비활성화"""
        logging.info("Localization mode disabled.")
        self.localization_mode.clear()

    def add_target_points(self):
        """목표 지점 추가"""
        valid_points = list(self.predefined_points.keys())
        logging.info(f"Available locations: {valid_points}")

        while True:
            first_target = input(f"Enter a target point from {valid_points}: ")
            if first_target in self.predefined_points:
                self.target_points.append(self.predefined_points[first_target])
                break
            else:
                logging.warning("Invalid location. Please select from the available options.")

        logging.info(f"Added target points: {self.target_points}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    controller = SLAMController()

    # 모드 선택
    mode = input("Select mode ('mapping' or 'localization'): ").strip().lower()
    if mode == "mapping":
        controller.enable_mapping()
    elif mode == "localization":
        controller.enable_localization()
    else:
        logging.error("Invalid mode selected. Exiting.")
        exit(1)

    # 경로 목표 설정
    controller.add_target_points()

    # 작업 시작
    controller.start_threads()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        controller.stop_threads()
