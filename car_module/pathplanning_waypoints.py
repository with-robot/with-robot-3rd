from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
import numpy as np
import time

# path planning & tracking & wheel vel. ctrl
class youBotPP:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simOMPL = self.client.require('simOMPL')
        self.run_flag = True
        self.not_first_here = False

    def init_coppelia(self):
        self.robotHandle = self.sim.getObject('/youBot')
        self.refHandle = self.sim.getObject('/youBot_ref')
        self.collVolumeHandle = self.sim.getObject('/youBot_coll')
        self.goalDummyHandles = self.sim.getObject('/goalDummy')
        self.waypoints = [self.sim.getObject('/bedroom1'),
                          self.sim.getObject('/bedroom2'),
                          self.sim.getObject('/toilet'),
                          self.sim.getObject('/entrance'),
                          self.sim.getObject('/dining'),
                          self.sim.getObject('/livingroom'),
                          self.sim.getObject('/balcony_init'),
                          self.sim.getObject('/balcony_end'),
                        ]        

        self.wheel_joints = [
            self.sim.getObject('/rollingJoint_fl'),  # front left
            self.sim.getObject('/rollingJoint_rl'),  # rear left
            self.sim.getObject('/rollingJoint_fr'),   # front right
            self.sim.getObject('/rollingJoint_rr'),  # rear right
        ]

        self.prev_forwback_vel = 0
        self.prev_side_vel = 0
        self.prev_rot_vel = 0

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
        # self.sim.setStepping(True)
        self.sim.startSimulation(True)
        while self.run_flag:
            for i in range(len(self.waypoints)):
                goalPos = self.sim.getObjectPosition(self.waypoints[i], -1)
                print(f'goal position : {goalPos}')
                try:
                    path = self.findPath(goalPos)
                    print(f'Find the path : waypoint{i}')
                except:
                    print('Fail to find path.')
                if path != None:
                    self.followPath(goalDummy=self.waypoints[i], path=path)
                    self.omni_wheel_control(0.0, 0.0, 0.0)
                    self.clear_path()
                print('move to another waypoint')
                time.sleep(2)
            print('check')
            self.omni_wheel_control(0.0, 0.0, 0.0)
            break
        print('robot reaches the goal position')
        self.sim.stopSimulation()

if __name__ == "__main__":
    controller = youBotPP()
    controller.init_coppelia()
    controller.run_coppelia()
