# not select the goals for path planning
# loop the waypoitns mode.

import threading
import time
import logging
from youBot import YouBot
from grid_mapping import MappingBot
from pathplanning import youBotPP
from mcl_coppelia import LocalizationBot


class SLAMController:
    def __init__(self):
        self.planning = youBotPP()
        self.mapping = MappingBot()
        self.localization = LocalizationBot()

        self.stop_event = threading.Event()  # 종료 이벤트
        self.mapping_mode = threading.Event()  # 맵핑 모드
        self.localization_mode = threading.Event()  # 로컬라이제이션 모드

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    slam_controller = SLAMController()

    try:
        # 사용자 입력으로 시작 모드 선택
        mode = None
        while mode not in ["mapping", "localization"]:
            mode = input("Select mode to start (mapping/localization): ").strip().lower()
            if mode == "mapping":
                slam_controller.enable_mapping()
                slam_controller.disable_localization()
            elif mode == "localization":
                slam_controller.enable_localization()
                slam_controller.disable_mapping()
            else:
                print("Invalid input. Please enter 'mapping' or 'localization'.")

        # 쓰레드 시작
        slam_controller.start_threads()

        # 메인 루프에서 모드 제어
        while True:
            user_input = input(
                "Enter command (m_on, m_off, l_on, l_off, quit): "
            ).strip().lower()
            if user_input == "m_on":
                slam_controller.enable_mapping()
            elif user_input == "m_off":
                slam_controller.disable_mapping()
            elif user_input == "l_on":
                slam_controller.enable_localization()
            elif user_input == "l_off":
                slam_controller.disable_localization()
            elif user_input == "quit":
                break
            else:
                print("Unknown command. Please enter a valid command.")

    except KeyboardInterrupt:
        logging.info("Interrupt received, stopping threads...")
    finally:
        slam_controller.stop_threads()
