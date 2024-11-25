import numpy as np
import matplotlib.pyplot as plt

from youBot import YouBot

class MappingBot(YouBot):
    def __init__(self):
        super().__init__()
        self.grid = Grid()

    def read_ref(self):
        x, y = self.sim.getObjectPosition(self.youBot_ref)[:2]
        # list eulerAngles = sim.getObjectOrientation(int objectHandle, 
        #                                     int relativeToObjectHandle = sim.handle_world)
        # eulerAngles: Euler angles [alpha beta gamma]
        '''
        - alpha = Roll
        - beta = Pitch
        - gamma = Yaw
        '''
        theta = self.sim.getObjectOrientation(self.youBot_ref)[2]
        return x, y, theta
    
    def run_step(self, count):
        # self.control_car()
        
        scan = self.read_lidars()
        print('yes')
        loc = self.read_ref()
        
        # update grid
        self.grid.update(loc, scan)

class Grid():
    def __init__(self):
        # grid 전체 배열 크기 100*100 으로 설정.
        self.grid = np.zeros((100, 100))
        # plot grid
        r = np.linspace(-5, 5, 101)  # -5부터 5까지 101개의 x 좌표 생성
        p = np.linspace(-5, 5, 101)  # -5부터 5까지 101개의 y 좌표 생성
        self.R, self.P = np.meshgrid(r, p)  # 2D 그리드 생성
        # plot object
        self.plt_objects = [None] * 15
        # scan theta
        self.delta = np.pi / 12     # 13개 라이다 -> 12개 간격으로 분해능.
        # -90도 기준으로 각 라이다별 스캔 각도 계산.
        self.scan_theta = np.array([- np.pi / 2 + self.delta * i for i in range(13)])
        self.boundary = np.pi / 2 + self.delta / 2      
        # min distance
        self.min_dist = (2 * (0.05**2)) ** 0.5  # 0.07071

    def update(self, loc, scan):
        self.mapping(loc, scan)
        self.save()
        self.visualize(loc, scan)

    def mapping(self, loc, scan):
        x, y, theta = loc       # youBot_ref (x,y,yaw)
        theta += np.pi/2
        # scan position
        # 라이다 위치는 ref position (로봇 중앙) 으로부터 앞으로 0.275 m 으로 조정.
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        # range
        dist = 2.25     # 스캔 센서 감지 최대 거리
        '''
        i_min, i_max, j_min, j_max는 그리드 맵에서 스캔 범위 내의 인덱스를 결정. 
        그리드의 해상도는 0.1 미터이고, 그리드 맵이 100 x 100 크기를 가지며, 맵의 중앙이 (50,50)에 해당
        '''
        # xy_resolution = 0.1
        i_min = max(0, int((rx - dist) // 0.1 + 50))
        i_max = min(99, int((rx + dist) // 0.1 + 50))
        j_min = max(0, int((ry - dist) // 0.1 + 50))
        j_max = min(99, int((ry + dist) // 0.1 + 50))
        # sub grid
        sub_grid = self.grid[j_min : j_max + 1, i_min : i_max + 1]  # grid[row, col] = grid[y_range, x_range]

        # x distance
        # 1) i_min 부터 i_max 까지 정수 배열 생성 -> 0.1 해상도 곱하여 미터 단위 변환
        # 2) 배열 원소에 +0.05 함으로써 각 그리드 셀의 중심 좌표 구함.
        # 3) 그리드 크기가 10*10 이므로 0으로부터 상대적 중앙을 구하기 위해 -5
        gx = np.arange(i_min, i_max +1) * 0.1 + 0.05 - 5     # numpy.arange([start, ] stop, [step, ] dtype=None)
        gx = np.repeat(gx.reshape(1,-1), sub_grid.shape[0], axis=0)     # gx 배열을 1*N 형태로 변환.
        dx = gx - rx    # 라이다 위치로부터 sub grid 각 셀의 x 방향 거리 추정.
        # y distance
        gy = np.arange(j_min, j_max + 1) * 0.1 + 0.05 - 5
        gy = np.repeat(gy.reshape(1,-1).T, sub_grid.shape[1], axis=1)
        dy = gy - ry
        # distance
        gd = (dx**2 + dy**2) ** 0.5     # 라이다 위치로부터 sub grid 각 셀 까지의 거리 추정.

        # theta diff
        # 각 지점과 로봇의 스캔 세서 사이의 각도 차이를 구하고, 로봇이 바라보는 방향 theta 와 비교해 각도 차이 dtheta 를 계산.
        gtheta = np.arccos(dx / gd) * ((dy > 0) * 2 -1)     # gtheta : global coord.origin 에 대한 광선의 각도
        dtheta = gtheta - theta                             # dtheta : 로봇의 yaw 를 고려한 광선의 상대 각도. 로봇 이동 방향 기준으로 계산된 광선의 각도.
        # 각도 범위 조정. (-pi ~ pi)
        while np.pi < np.max(dtheta):
            dtheta -= (np.pi < dtheta) * 2 * np.pi
        while np.min(dtheta) < -np.pi:
            dtheta += (dtheta < -np.pi) * 2 * np.pi
        
        # inverse sensor model
        '''
        Return values of 'sim.readProximitySensor'
        :   res = detection state (0 or 1)
            dist = distance to th detected point
            point = array of 3 numbers indicating the relative coord. of the detected point
            obj = handle of the object that was detected
            n = normal vector of the detected surface relative to the snesor reference frame. 
        '''
        # res 는 센서가 장애물을 감지했는지 여부, dist 는 장애물과의 거리
        for i in range(13):
            res, dist, _, _, _ = scan[i]
            if res == 0:        # 장애물이 없는 경우 -> 해당 방향에서 특정 거리 내에 있는 셀들을 자유 공간으로 간주. free area
                area = (
                    (gd <= 2.25)    # sub gird 각 셀과 로봇 센서 사이의 거리가 2.25 미터 이하인지 확인.
                    * (-self.boundary + self.delta * i <= dtheta)       # i 번째 스캔 데이터가 커버하는 각도 범위 내에 있는 그리드 셀만을 업데이트 대상으로.
                    * (dtheta <= -self.boundary + self.delta * (i +1))
                )
                sub_grid[area] -= 0.5   # 위 조건을 만족하는 area 에서 그리드 셀의 값을 0.5 만큼 감소 -> 자유공간 가능성.
            else:       # 장애물 감지.
                dist = min(2.25, dist)      # dist 재정의.
                detect_area = (
                    (np.abs(gd - dist) < self.min_dist)     # 그리드 상의 각 지점과 로봇 센서 사이의 거리가 탐지된 거리와 매우 근접한지 여부 확인. 기준 값 : 0.07071
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[detect_area] += 0.5    # 그리드 값을 0.5만큼 증가 -> 장애물 있을 가능성 큼.
                free_area = (
                    (gd <= dist - self.min_dist)
                    * (-self.boundary + self.delta * i <= dtheta)
                    * (dtheta <= -self.boundary + self.delta * (i + 1))
                )
                sub_grid[free_area] -= 0.5      # 장애물 있을 가능성 존재 -> 자유 공간 셀 가중치 감소.
        np.clip(self.grid, -5, 5, out=self.grid)        # 마지막으로 그리드의 모든 셀 값이 -5에서 5 사이로 제한.

    def save(self):
        with open("/home/oh/my_coppeliasim/with-robot-3rd/car_module/mapping.npy", "wb") as f:
            np.save(f, self.grid)

    def visualize(self, loc, scan):
        x, y, theta = loc
        theta += np.pi/2
        # clear object
        for object in self.plt_objects:
            if object:
                object.remove()
        # grid
        grid = -self.grid + 5
        self.plt_objects[0] = plt.pcolor(self.R, self.P, grid, cmap="gray")
        # robot
        (self.plt_objects[1],) = plt.plot(
            x, y, color="green", marker="o", markersize=10
        )
        # scan
        rx = x + 0.275 * np.cos(theta)
        ry = y + 0.275 * np.sin(theta)
        for i, data in enumerate(scan):
            res, dist, _, _, _ = data  # res, dist, point, obj, n
            style = "--r" if res == 1 else "--b"
            dist = dist if res == 1 else 2.20

            ti = theta + self.scan_theta[i]
            xi = rx + dist * np.cos(ti)
            yi = ry + dist * np.sin(ti)
            (self.plt_objects[2 + i],) = plt.plot([rx, xi], [ry, yi], style)

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.gca().set_aspect("equal")
        plt.pause(0.001)

if __name__ == "__main__":
    client = MappingBot()
    client.init_coppelia()
    client.run_coppelia()