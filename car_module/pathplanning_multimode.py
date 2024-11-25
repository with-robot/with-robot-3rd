from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

class youBotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

        self.mode = "mapping"
        self.single_goal = None

        self.predefined_points = {
            "bedroom1": '/bedroom1',
            "bedroom2": '/bedroom2',
            "toilet": '/toilet',
            "entrance": '/entrance',
            "dining": '/dining',
            "livingroom": '/livingroom',
            "balcony_init": '/balcony_init',
            "balcony_end": '/balcony_end',
        }

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        self.goalDummyHandles = self.sim.getObject('/goalDummy')
        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),
            self.sim.getObject('/rollingJoint_rl'),
            self.sim.getObject('/rollingJoint_fr'),
            self.sim.getObject('/rollingJoint_rr'),
        ]

        self.waypoints = [
            self.sim.getObject(self.predefined_points[point])
            for point in self.predefined_points.keys()
        ]

        # 초기화
        self.prev_forwback_vel = 0
        self.prev_side_vel = 0
        self.prev_rot_vel = 0
        self.line_container = None

        self.p_parm = 50 #20
        self.max_v = 10
        self.p_parm_rot = 10 #10
        self.max_v_rot = 3
        self.accel_f = 0.35


        self.robotObstaclesCollection = self.sim.createCollection(0)
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_all, -1, 0)     # "-1" means world?
        self.sim.addItemToCollection(self.robotObstaclesCollection, self.sim.handle_tree, self.robotHandle, 1)
        self.collPairs = [self.collVolumeHandle, self.robotObstaclesCollection]

        self.search_range = 10
        self.search_algo = self.simOMPL.Algorithm.BiTRRT
        self.search_duration = 0.1

        self.display_collision_free_nodes = True
        self.show_real_target = True
        self.show_track_pos = True
        self.line_container = None

    def set_mode(self, mode, goal=None):
        """Set the planning mode and optionally a single goal."""
        self.mode = mode
        if mode == "localization" and goal:
            self.single_goal = goal

    def visualizePath(self, path):
        if self.line_container is None:
            self.line_container = self.sim.addDrawingObject(self.sim.drawing_lines, 3, 0, -1, 99999, [0.2, 0.2, 0.2])
        self.sim.addDrawingObjectItem(self.line_container, None)

        if path:
            for i in range(1, len(path)//2):
                line_data = [path[2*i-2], path[2*i -1], 0.001, path[2*i], path[2*i+1], 0.001]
                self.sim.addDrawingObjectItem(self.line_container, line_data)

    def findPath(self, targetPos):
        path = None
        while not path:
            task = self.simOMPL.createTask(f't')
            self.simOMPL.setAlgorithm(task, self.search_algo)
            startPos = self.sim.getObjectPosition(self.refHandle, -1)
            # rstatespace 탐색 결과에 대한 예외 처리 힐요.
            ss = [self.simOMPL.createStateSpace('2d', self.simOMPL.StateSpaceType.position2d, self.collVolumeHandle,
                                                [startPos[0] - self.search_range, startPos[1] - self.search_range],
                                                [startPos[0] + self.search_range, startPos[1] + self.search_range], 1)]
            self.simOMPL.setStateSpace(task, ss)
            self.simOMPL.setCollisionPairs(task, self.collPairs)
            self.simOMPL.setStartState(task, startPos[:2])
            self.simOMPL.setGoalState(task, targetPos[:2])
            self.simOMPL.setStateValidityCheckingResolution(task, 0.01)
            self.simOMPL.setup(task)
            print('10')
            if self.simOMPL.solve(task, self.search_duration):
                print('11')
                self.simOMPL.simplifyPath(task, self.search_duration)       # search_duration 반경 넘어가면 simOMPL BiRRT 작동 에러.
                path = self.simOMPL.getPath(task)
                self.visualizePath(path)
            else: 
                # path
                self.omni_wheel_control(0.2,0.2,0.2)
            time.sleep(0.01)
        return path
    
    def followPath(self, goalDummy, path=None):
        if path:
            path_3d = []
            for i in range(0, len(path)//2):
                path_3d.extend([path[2*i], path[2*i+1], 0.0])
            prev_dist = 0
            track_pos_container = self.sim.addDrawingObject(self.sim.drawing_spherepoints | self.sim.drawing_cyclic, 0.02, 0, -1, 1, [1, 0, 1])
            while True:
                currPos = self.sim.getObjectPosition(self.refHandle, -1)

                pathLength, totalDist = self.sim.getPathLengths(path_3d, 3)
    
                closet_dist = self.sim.getClosestPosOnPath(path_3d, pathLength, currPos)

                if closet_dist <= prev_dist:
                    closet_dist += totalDist / 200
                prev_dist = closet_dist

                tartgetPoint = self.sim.getPathInterpolatedConfig(path_3d, pathLength, closet_dist)
                self.sim.addDrawingObjectItem(track_pos_container, tartgetPoint)
                
                m = self.sim.getObjectMatrix(self.refHandle, -1)
                m_inv = self.sim.getMatrixInverse(m)

                rel_p = self.sim.multiplyVector(m_inv, tartgetPoint)
                rel_o = math.atan2(rel_p[1], rel_p[0]) - math.pi/2      # yaw 조절하는 부분.

                forwback_vel = rel_p[1] * self.p_parm
                side_vel = rel_p[0] * self.p_parm
                v = (forwback_vel**2 + side_vel**2)**0.5
                if v > self.max_v:
                    forwback_vel *= self.max_v / v
                    side_vel *= self.max_v / v

                rot_vel = -rel_o * self.p_parm_rot
                if abs(rot_vel) > self.max_v_rot :
                    rot_vel = self.max_v_rot * rot_vel / abs(rot_vel)

                df = forwback_vel - self.prev_forwback_vel
                ds = side_vel - self.prev_side_vel
                dr = rot_vel - self.prev_rot_vel

                if abs(df) > self.max_v * self.accel_f:
                    df = self.max_v * self.accel_f * df / abs(df)
                if abs(ds) > self.max_v * self.accel_f:
                    ds = self.max_v * self.accel_f * ds / abs(ds)
                if abs(dr) > self.max_v_rot * self.accel_f:
                    dr = self.max_v_rot * self.accel_f * dr / abs(dr)


                forwback_vel = self.prev_forwback_vel + df
                side_vel = self.prev_side_vel + ds 
                rot_vel = self.prev_rot_vel + dr 

                self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
                self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
                
                self.prev_forwback_vel = forwback_vel
                self.prev_side_vel = side_vel
                self.prev_rot_vel = rot_vel

                if np.linalg.norm(np.array(self.sim.getObjectPosition(goalDummy, -1)) -
                                  np.array(self.sim.getObjectPosition(self.refHandle, -1))) < 0.6:
                    
                    self.sim.removeDrawingObject(track_pos_container)
                    break
                self.sim.step()
                # time.sleep(0.001)

    def omni_wheel_control(self, forwback_vel, side_vel, rot_vel):
        self.sim.setJointTargetVelocity(self.wheel_joints[0], -forwback_vel - side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[1], -forwback_vel + side_vel - rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[2], -forwback_vel + side_vel + rot_vel)
        self.sim.setJointTargetVelocity(self.wheel_joints[3], -forwback_vel - side_vel + rot_vel)
   
    def clear_path(self):
        self.sim.removeDrawingObject(self.line_container)
        self.line_container = None

    def run_coppelia(self):
        """Run path planning based on the selected mode."""
        self.sim.startSimulation(True)
        if self.mode == "mapping":
            while self.run_flag:
                for waypoint in self.waypoints:
                    goalPos = self.sim.getObjectPosition(waypoint, -1)
                    print(f"Goal position: {goalPos}")
                    try:
                        path = self.findPath(goalPos)
                        print(f"Found path to waypoint: {waypoint}")
                    except:
                        print("Failed to find path.")
                        continue
                    if path:
                        self.followPath(goalDummy=waypoint, path=path)
                        self.omni_wheel_control(0.0, 0.0, 0.0)
                        self.clear_path()
                    time.sleep(2)  # Delay between waypoints
                self.omni_wheel_control(0.0, 0.0, 0.0)
                break
        elif self.mode == "localization" and self.single_goal:
            print(f"Planning to single goal: {self.single_goal}")
            try:
                path = self.findPath(self.single_goal)
                if path:
                    self.followPath(goalDummy=None, path=path)
                    self.omni_wheel_control(0.0, 0.0, 0.0)
                    self.clear_path()
            except:
                print("Failed to find path to the goal.")
        self.sim.stopSimulation()

    def run_waypoints(self):
        """여러 지점 순찰 모드"""
        for waypoint in self.waypoints:
            goalPos = self.sim.getObjectPosition(waypoint, -1)
            print(f"Waypoint goal position: {goalPos}")
            path = self.findPath(goalPos)
            if path:
                self.followPath(goalDummy=waypoint, path=path)
                self.omni_wheel_control(0.0, 0.0, 0.0)
                self.clear_path()
            else:
                print("Path not found for waypoint.")
            time.sleep(2)

    def run_single_goal(self, goal_name):
        """단일 목표 지점 이동 모드"""
        if goal_name not in self.predefined_points:
            print(f"Invalid goal: {goal_name}. Stopping simulation.")
            return

        goalHandle = self.sim.getObject(self.predefined_points[goal_name])
        goalPos = self.sim.getObjectPosition(goalHandle, -1)
        print(f"Single goal position: {goalPos}")

        path = self.findPath(goalPos)
        if path:
            self.followPath(goalDummy=goalHandle, path=path)
            self.omni_wheel_control(0.0, 0.0, 0.0)
            self.clear_path()
        else:
            print("Path not found for single goal.")

    # (기존 findPath, followPath, omni_wheel_control, clear_path 메서드는 변경 없음)

if __name__ == "__main__":
    controller = youBotPP()
    controller.init_coppelia()

    # 사용자 입력으로 모드 선택
    mode = input("Select mode ('single_goal' or 'waypoints'): ").strip().lower()

    if mode == "single_goal":
        single_goal = input("Enter the goal name (e.g., 'bedroom1', 'toilet'): ").strip()
        controller.run_coppelia(mode="single_goal_mode", single_goal=single_goal)
    elif mode == "waypoints":
        controller.run_coppelia(mode="waypoints_mode")
    else:
        print("Invalid mode selected. Exiting.")
