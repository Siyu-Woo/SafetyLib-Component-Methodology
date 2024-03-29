from typing import List, Tuple, Dict, Callable
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

Point = Tuple[float, float]
Ori = Tuple[float, float]

# global delta_s=0.01

# --------------------------------点列计算方法-----------------------------------#
# 基于两点生成直线参考线上的点坐标
# start_point=参考线的起始点坐标, end_point=参考线的终点坐标, delta_s=采样步长
def generate_reference_line_points(start_point, end_point, delta_s):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = (dx ** 2 + dy ** 2) ** 0.5

    x_func = lambda s: start_point[0] + (dx / length) * s
    y_func = lambda s: start_point[1] + (dy / length) * s

    points = []
    s = 0
    while s <= length:
        x = x_func(s)
        y = y_func(s)
        points.append((x, y))
        s += delta_s
    if s - delta_s < length:
        x = x_func(length)
        y = y_func(length)
        points.append((x, y))

    return points, length

# 生成直道车道线点列字典
# refline_points=参考线上的点列, left_lane_count=左侧车道数量, right_lane_count=右侧车道数量, delta_s=采样间隔
# lanes=车道字典（左负右正）->车道宽度序列的长度需和参考线上点列长度一致（即颗粒度一样）
def generate_lane_lines(refline_points, left_lane_count, right_lane_count, delta_s, lanes):
    lane_lines = {}
    cum_width = {}
    for key in lanes.keys():
        lane_lines[key] = []
        cum_width[key] = []

    # Ensure all lanes are present in lane_lines
    for i in range(-left_lane_count, right_lane_count + 1):
        lane_lines[i] = []

    for i in range(1, len(refline_points) + 1):
        s = i * delta_s
        left_cum_width = 0
        for j in range(1, left_lane_count + 1):
            left_cum_width += get_lane_width(lanes[-j], s, delta_s)
            cum_width[-j].append(left_cum_width)
        right_cum_width = 0
        for j in range(1, right_lane_count + 1):
            right_cum_width += get_lane_width(lanes[-j], s, delta_s)
            cum_width[j].append(right_cum_width)

    for lane_id, lane in lanes.items():
        lane_points = []
        for i in range(1, len(refline_points)):
            current_point = refline_points[i - 1]
            next_point = refline_points[i]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            if lane_id > 0:
                norm_vector = (-dy, dx)
            else:
                norm_vector = (dy, -dx)
            norm_length = math.sqrt(norm_vector[0] ** 2 + norm_vector[1] ** 2)
            unit_vector = (norm_vector[0] / norm_length, norm_vector[1] / norm_length)

            new_x = current_point[0] + unit_vector[0] * cum_width[lane_id][i]
            new_y = current_point[1] + unit_vector[1] * cum_width[lane_id][i]
            new_point = (new_x, new_y)
            lane_points.append(new_point)

        current_point = refline_points[len(refline_points) - 1]
        prev_point = refline_points[len(refline_points) - 2]
        dx = current_point[0] - prev_point[0]
        dy = current_point[1] - prev_point[1]
        if lane_id > 0:
            norm_vector = (-dy, dx)  # Right lane: rotate 90 degrees clockwise
        else:
            norm_vector = (dy, -dx)  # Left lane: rotate 90 degrees counterclockwise
        norm_length = math.sqrt(norm_vector[0] ** 2 + norm_vector[1] ** 2)
        unit_vector = (norm_vector[0] / norm_length, norm_vector[1] / norm_length)

        new_x = current_point[0] + unit_vector[0] * cum_width[lane_id][len(refline_points) - 1]
        new_y = current_point[1] + unit_vector[1] * cum_width[lane_id][len(refline_points) - 1]
        new_point = (new_x, new_y)
        lane_points.append(new_point)

        lane_lines[lane_id] = lane_points

    return lane_lines

# 提取直道道路边界线（默认道路最外侧车道线即为道路边界线）
# lane_lines=车道字典, refline_points=参考线点列, left_lane_count=左侧车道数量, right_lane_count=右侧车道数量
def generate_boundary_lines(lane_lines, refline_points, left_lane_count, right_lane_count):
    boundary_lines = {'left': [], 'right': []}

    if left_lane_count == 0:
        boundary_lines['left'] = refline_points
    else:
        boundary_lines['left'] = lane_lines[-left_lane_count]

    if right_lane_count == 0:
        boundary_lines['right'] = refline_points
    else:
        boundary_lines['right'] = lane_lines[right_lane_count]

    return boundary_lines

# 根据匝道的起点和终点生成匝道的参考线点坐标
#start_point=匝道的起点坐标, end_point=匝道的终点坐标, delta_s=参考线的采样步长
def generate_ramp_reference_line(start_point: Point, end_point: Point, delta_s: float) -> Tuple[List[Point], float]:
    refline_points, length = generate_reference_line_points(start_point, end_point, delta_s)
    return refline_points, length

# 根据匝道的参考线和车道信息生成匝道的车道线点列（含重叠部分）
# refline_points=匝道的参考线上的点列, delta_s=采样步长, left_lane_count=左侧车道数量, right_lane_count=右侧车道数量, lanes=匝道的车道信息
def generate_ramp_lane_lines(refline_points: List[Point], delta_s: float, left_lane_count: int, right_lane_count: int, lanes: Dict[int, Dict]) -> Dict[int, List[Point]]:
    lane_lines = {}
    cum_width = {}
    for key in lanes.keys():
        lane_lines[key] = []
        cum_width[key] = []

    # 确保所有车道都在车道线上
    for i in range(-left_lane_count, right_lane_count + 1):
        lane_lines[i] = []

    for i in range(1, len(refline_points) + 1):
        s = i * delta_s
        left_cum_width = 0
        for j in range(1, left_lane_count + 1):
            left_cum_width += get_lane_width(lanes[-j], s, delta_s)
            cum_width[-j].append(left_cum_width)
        right_cum_width = 0
        for j in range(1, right_lane_count + 1):
            right_cum_width += get_lane_width(lanes[j], s, delta_s)
            cum_width[j].append(right_cum_width)

    for lane_id, lane in lanes.items():
        lane_points = []
        for i in range(1, len(refline_points)):
            current_point = refline_points[i - 1]
            next_point = refline_points[i]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            if lane_id > 0:
                norm_vector = (-dy, dx)
            else:
                norm_vector = (dy, -dx)
            norm_length = math.sqrt(norm_vector[0] ** 2 + norm_vector[1] ** 2)
            unit_vector = (norm_vector[0] / norm_length, norm_vector[1] / norm_length)

            new_x = current_point[0] + unit_vector[0] * cum_width[lane_id][i]
            new_y = current_point[1] + unit_vector[1] * cum_width[lane_id][i]
            new_point = (new_x, new_y)
            lane_points.append(new_point)

        current_point = refline_points[len(refline_points) - 1]
        prev_point = refline_points[len(refline_points) - 2]
        dx = current_point[0] - prev_point[0]
        dy = current_point[1] - prev_point[1]
        if lane_id > 0:
            norm_vector = (-dy, dx)
        else:
            norm_vector = (dy, -dx)
        norm_length = math.sqrt(norm_vector[0] ** 2 + norm_vector[1] ** 2)
        unit_vector = (norm_vector[0] / norm_length, norm_vector[1] / norm_length)

        new_x = current_point[0] + unit_vector[0] * cum_width[lane_id][len(refline_points) - 1]
        new_y = current_point[1] + unit_vector[1] * cum_width[lane_id][len(refline_points) - 1]
        new_point = (new_x, new_y)
        lane_points.append(new_point)

        lane_lines[lane_id] = lane_points

    return lane_lines

# 根据匝道的车道线和参考线生成匝道的边界线点列（含重叠部分）
# lane_lines=匝道的车道线, refline_points=匝道的参考线上的点列, left_lane_count=左侧车道数量, right_lane_count=右侧车道数量
def generate_ramp_boundary_lines(lane_lines: Dict[int, List[Point]], refline_points: List[Point], left_lane_count: int, right_lane_count: int) -> Dict[str, List[Point]]:
    boundary_lines = {'left': [], 'right': []}

    if left_lane_count == 0:
        boundary_lines['left'] = refline_points
    else:
        boundary_lines['left'] = lane_lines[-left_lane_count]

    if right_lane_count == 0:
        boundary_lines['right'] = refline_points
    else:
        boundary_lines['right'] = lane_lines[right_lane_count]

    return boundary_lines

# （工具函数）算由端点A-B和C-D定义的两条线段的交点
def calculate_intersection(A, B, C, D):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    if intersect(A, B, C, D):
        x1, y1 = A
        x2, y2 = B
        x3, y3 = C
        x4, y4 = D

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denominator == 0:
            return None

        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

        epsilon = 1e-3
        if (min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon and
                min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon and
                min(x3, x4) - epsilon <= x <= max(x3, x4) + epsilon and
                min(y3, y4) - epsilon <= y <= max(y3, y4) + epsilon):
            return x, y
        else:
            return None
    else:
        return None
    
# 计算匝道边界线与主路边界线的交点
# 参数:
#     - ramp_boundary_lines (Dict[str, List[Point]]): 匝道的边界线。
#     - main_road_boundary_lines (Dict[str, List[Point]]): 主路的边界线。
def calculate_intersection_points(ramp_boundary_lines: Dict[str, List[Tuple[float, float]]], main_road_boundary_lines: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    ramp_left_boundary = np.array(ramp_boundary_lines['left'])
    ramp_right_boundary = np.array(ramp_boundary_lines['right'])
    main_road_left_boundary = np.array(main_road_boundary_lines['left'])
    main_road_right_boundary = np.array(main_road_boundary_lines['right'])

    intersection_points = []
    for line1, line2 in [(ramp_left_boundary, main_road_left_boundary),
                         (ramp_left_boundary, main_road_right_boundary),
                         (ramp_right_boundary, main_road_left_boundary),
                         (ramp_right_boundary, main_road_right_boundary)]:
        for i in range(len(line1) - 1):
            for j in range(len(line2) - 1):
                point = calculate_intersection(line1[i].tolist(), line1[i + 1].tolist(), line2[j].tolist(), line2[j + 1].tolist())
                if point:
                    intersection_points.append(point)
    print(intersection_points)
    return intersection_points

def calculate_intersection_points_lane(ramp_lane_lines: Dict[int, List[Tuple[float, float]]], main_road_boundary_lines: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    intersection_points = []
    for lane_points in ramp_lane_lines.values():
        for point1, point2 in zip(lane_points[:-1], lane_points[1:]):
            for side in ['left', 'right']:
                for main_road_point1, main_road_point2 in zip(main_road_boundary_lines[side][:-1], main_road_boundary_lines[side][1:]):
                    point = calculate_intersection(point1, point2, main_road_point1, main_road_point2)
                    if point:
                        intersection_points.append(point)
    return intersection_points

# 基于道路向量确定保留点列
def remove_points_using_vector_rule(points_on_ramp: List[Tuple[float, float]], intersection_point: Tuple[float, float], main_road_start_point: Tuple[float, float], main_road_end_point: Tuple[float, float]) -> List[Tuple[float, float]]:
    # 计算主道路的向量
    main_road_vector = (main_road_end_point[0] - main_road_start_point[0], main_road_end_point[1] - main_road_start_point[1])

    remaining_points = []

    for point in points_on_ramp:
        ramp_point_vector = (point[0] - intersection_point[0], point[1] - intersection_point[1])

        cross_product = main_road_vector[0] * ramp_point_vector[1] - main_road_vector[1] * ramp_point_vector[0]

        # 如果叉乘大于0，则保留该点
        if cross_product > 0:
            remaining_points.append(point)

    return remaining_points

# 移除匝道边界线上的点
def remove_points_at_intersection(ramp_boundary_lines: Dict[str, List[Tuple[float, float]]], intersection_points: List[Tuple[float, float]], main_road_boundary_lines: Dict[str, List[Tuple[float, float]]]):
    for side in ['left', 'right']:
        for intersection_point in intersection_points:
            if side == 'left':
                remaining_points = remove_points_using_vector_rule(ramp_boundary_lines['left'], intersection_point, main_road_boundary_lines['left'][0], main_road_boundary_lines['left'][-1])
                ramp_boundary_lines['left'] = remaining_points
            elif side == 'right':
                remaining_points = remove_points_using_vector_rule(ramp_boundary_lines['right'], intersection_point, main_road_boundary_lines['right'][0], main_road_boundary_lines['right'][-1])
                ramp_boundary_lines['right'] = remaining_points

# 移除匝道车道线上的点
def remove_points_at_intersection_lane(ramp_lane_lines: Dict[int, List[Tuple[float, float]]], intersection_points: List[Tuple[float, float]], main_road_boundary_lines: Dict[str, List[Tuple[float, float]]]):
    for lane_id, lane_points in ramp_lane_lines.items():
        remaining_points = []
        for intersection_point in intersection_points:
            remaining_points += remove_points_using_vector_rule(lane_points, intersection_point,
                                                                main_road_boundary_lines['left'][0],
                                                                main_road_boundary_lines['left'][-1])
        ramp_lane_lines[lane_id] = remaining_points

# 移除主路边界线与匝道交汇处的点
def remove_points_between_boundaries(main_road_boundary_lines: Dict[str, List[Tuple[float, float]]], intersection_points: List[Tuple[float, float]]):
    # 定义误差范围
    epsilon = 1e-6

    # 对主路边界线进行处理
    for side in ['left', 'right']:
        remaining_points = []
        for point in main_road_boundary_lines[side]:
            # 判断点是否在交点之间，若在，则跳过
            is_between = False
            for intersection_point in intersection_points:
                # 获取交点的 x 坐标范围
                intersection_x_1 = intersection_points[0][0]
                intersection_x_2 = intersection_points[1][0]
                # 获取交点的 y 坐标范围
                intersection_y_1 = intersection_points[0][1]
                intersection_y_2 = intersection_points[1][1]
                # 判断点是否在交点之间，考虑误差范围
                if min(intersection_x_1, intersection_x_2) - epsilon <= point[0] <= max(intersection_x_1, intersection_x_2) + epsilon and \
                   min(intersection_y_1, intersection_y_2) - epsilon <= point[1] <= max(intersection_y_1, intersection_y_2) + epsilon:
                    is_between = True
                    break
            if not is_between:
                remaining_points.append(point)
        main_road_boundary_lines[side] = remaining_points



# --------------------------------宽度计算方法-----------------------------------#
# 获取长度s处的车道宽度
# lane=道路序列（仅一个车道）, s=沿着车道长度的位置,delta_s=宽度采样颗粒度
def get_lane_width(lane, s, delta_s=0.1):
    index = int(s/delta_s)
    samples = lane['width_samples']
    if 0 <= index < len(samples):
        return samples[index]
    elif index >= len(samples):
        return samples[-1]
    else:
        return samples[0]

# 调整指定车道宽度（变宽/变窄）
# lane_sequence=车道序列（需要调整的对象）, LaneID=需要处理处理的车道id, start_s=开始调整位置, end_s=结束调整位置, end_width=目标宽度, delta_s=采样步长
def adjust_lane_width(lane_sequence, LaneID, start_s, end_s, end_width, delta_s=0.1):
    # 获取车道宽度的初始值和结束值
    start_width = get_lane_width(lane_sequence[LaneID], start_s, delta_s)
    length = len(lane_sequence[LaneID]['width_samples']) * delta_s

    # 计算起始和结束的索引位置
    start_index = int(start_s / delta_s)
    end_index = min(int(end_s / delta_s), len(lane_sequence[LaneID]['width_samples']) - 1)

    # 生成宽度变化的序列
    width_change = (end_width - start_width) / (end_index - start_index)
    new_width_samples = [start_width] * start_index  # 保持起点之前的宽度不变
    new_width_samples.extend(start_width + i * width_change for i in range(end_index - start_index + 1))

    # 处理剩余部分为终止宽度
    remaining_length = length - (end_index - start_index) * delta_s
    remaining_samples = int(remaining_length / delta_s)
    new_width_samples.extend([end_width] * remaining_samples)

    return new_width_samples

# 计算道路左侧车道总宽度
# lanes=车道字典, left_lane_count=左侧车道数量, length=道路长度
def _calculate_road_left_width(lanes, left_lane_count, length):
    if left_lane_count == 0:
        return lambda s: [0] * len(lanes[-1]['width_samples'])
    elif left_lane_count == 1:
        return lanes[-1]['width_samples']
    else:
        lane_widths = []
        for i in range(len(lanes[-1]['width_samples'])):
            s = i / (len(lanes[-1]['width_samples']) - 1) * length
            total_width = sum(get_lane_width(lane, s) for lane in lanes.values() if lane['id'] < 0)
            lane_widths.append(total_width)
        return lane_widths

# 计算道路右侧车道总宽度
# lanes=车道字典, right_lane_count=右侧车道数量, length=道路长度
def _calculate_road_right_width(lanes, right_lane_count, length):
    if right_lane_count == 0:
        return lambda s: [0] * len(lanes[1]['width_samples'])
    elif right_lane_count == 1:
        return lanes[1]['width_samples']
    else:
        lane_widths = []
        for i in range(len(lanes[1]['width_samples'])):
            s = i / (len(lanes[1]['width_samples']) - 1) * length
            total_width = sum(get_lane_width(lane, s) for lane in lanes.values() if lane['id'] > 0)
            lane_widths.append(total_width)
        return lane_widths


# ----------------------------验证样本生成---------------------------------#
# 创建单个车道信息字典
# id=车道id, default_width=默认车道宽度（起始车道宽度）, end_width=终止车道宽度, length=车道长度
def create_lane(id=0, default_width=1, end_width=1, length=1):
    return {
        'id': id,
        'default_width': default_width,
        'width_samples': generate_width_samples(default_width, end_width, length=length)
    }

# 生成车道宽度序列样本
# start_width=起点坐标, end_width=终点坐标, length=车道长度
def generate_width_samples(start_width, end_width=3.5, length=1):
    samples = []
    num_samples = int(length / 0.1) + 1
    if num_samples == 1:  # 如果只有一个采样点，直接返回起始宽度
        return [start_width]

    # 计算宽度的变化量
    width_change = (end_width - start_width) / (num_samples - 1)

    # 生成采样点
    for _ in range(num_samples):
        samples.append(start_width)
        start_width += width_change

    return samples

#生成道路（含多个车道，键值为id-车道信息）
# default_lane_count=车道数量, default_width=默认车道宽度（起始车道宽度）, end_width=终止车道宽度, length=车道长度
def _generate_lanes(default_lane_count, default_width, end_width, length=2):
    lanes = {}
    for i in range(-default_lane_count, default_lane_count + 1):
        if i == 0:
            continue
        lanes[i] = create_lane(i, default_width, end_width, length=length)
    return lanes

# 根据给定参数生成匝道的起点和终点
# start_point=主路的起点坐标, end_point=主路的终点坐标, s=主路上匝道的起始位置, angle=匝道相对于主路的角度, ramp_length=匝道的长度
def generate_ramp_points(start_point: Tuple[float, float], end_point: Tuple[float, float], s: float, angle: float,
                         ramp_length: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # 计算主路的长度
    main_road_length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)

    # 计算匝道起点的坐标
    ramp_start_point = (start_point[0] + s * (end_point[0] - start_point[0]) / main_road_length,
                        start_point[1] + s * (end_point[1] - start_point[1]) / main_road_length)

    # 计算匝道终点的坐标
    dx = ramp_length * math.cos(math.radians(angle))
    dy = ramp_length * math.sin(math.radians(angle))
    ramp_end_point = (ramp_start_point[0] + dx, ramp_start_point[1] + dy)

    return ramp_start_point, ramp_end_point

# ------------------------画图函数----------------
# 描出数据集中点列
def plot_road(road, lane_lines, boundary_lines):
    refline_func = road['refline_func']
    print(road['refline_func'])
    s_values = np.linspace(0, road['length'], 100)
    refline_points = [(refline_func[0](s), refline_func[1](s)) for s in s_values]
    plt.plot(*zip(*refline_points), color='black', linestyle='--', label='Reference Line')

    for lane_id, points in lane_lines.items():
        plt.plot(*zip(*points), label=f'Lane {lane_id}')

    plt.plot(*zip(*boundary_lines['left']), color='red', label='Left Boundary')
    plt.plot(*zip(*boundary_lines['right']), color='blue', label='Right Boundary')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Road with Lane Markings')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

  # 调用可视化函数
def visualize_road(main_road_refline_points, ramp_refline_points, main_road_lane_lines, ramp_lane_lines,
                   main_road_boundary_lines, ramp_boundary_lines):
    fig, ax = plt.subplots()

    # 绘制主路参考线点
    main_road_x, main_road_y = zip(*main_road_refline_points)
    ax.scatter(main_road_x, main_road_y, color='black', label='Main Road Reference Line', s=0.1)

    # 绘制匝道参考线点
    ramp_x, ramp_y = zip(*ramp_refline_points)
    ax.scatter(ramp_x, ramp_y, color='blue', label='Ramp Reference Line', s=0.1)

    # 绘制主路车道线点
    for lane_line in main_road_lane_lines.values():
        if lane_line:  # 检查车道线是否为空
            lane_x, lane_y = zip(*lane_line)
            ax.scatter(lane_x, lane_y, color='black', s=0.1)

    # 绘制匝道车道线点
    for lane_line in ramp_lane_lines.values():
        if lane_line:  # 检查车道线是否为空
            lane_x, lane_y = zip(*lane_line)
            ax.scatter(lane_x, lane_y, color='blue', s=0.1)

    # 绘制主路边界线点
    for boundary_line in main_road_boundary_lines.values():
        boundary_x, boundary_y = zip(*boundary_line)
        ax.scatter(boundary_x, boundary_y, color='red', s=0.1)

    # 绘制匝道边界线点
    for boundary_line in ramp_boundary_lines.values():
        boundary_x, boundary_y = zip(*boundary_line)
        ax.scatter(boundary_x, boundary_y, color='red', s=0.1)

    # 添加图例
    custom_lines = [Line2D([0], [0], marker='o', color='black', linestyle='', markersize=0.1),
                    Line2D([0], [0], marker='o', color='black', linestyle='', markersize=0.1),
                    Line2D([0], [0], marker='o', color='blue', linestyle='', markersize=0.1),
                    Line2D([0], [0], marker='o', color='blue', linestyle='', markersize=0.1)]
    ax.legend(custom_lines, ['Main Road Reference Line', 'Main Road Lane Lines',
                             'Ramp Reference Line', 'Ramp Lane Lines'])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 显示图形
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Road Layout')
    plt.grid(True)
    plt.show()

# ——————————————————————————————————————验证——————————————————————————————————————
def generate_road_example():
    # 创建空列表来收集主路和匝道的所有点
    all_points = []

    # 定义主路的几何参数
    main_road_start_point = (0, 0)
    main_road_end_point = (100, 0)
    default_lane_count = 3
    default_lane_width = 3
    end_width = 3
    delta_s_main_road = 0.1  # 主路参考线采样步长

    # 生成主路的参考线和车道信息
    main_road_refline_points, main_road_length = generate_reference_line_points(main_road_start_point,
                                                                                main_road_end_point,
                                                                                delta_s_main_road)
    main_road_lanes = _generate_lanes(default_lane_count, default_lane_width, end_width, length=main_road_length)
    LaneID=1
#   print(main_road_lanes[LaneID]['width_samples'])
    main_road_lanes[LaneID]['width_samples']=adjust_lane_width(main_road_lanes, LaneID, 20, 50, 4, delta_s=0.1)

    # 定义匝道的几何参数
    ramp_length = 100
    ramp_angle = 30  # 匝道相对于主路的角度（单位：度）
    delta_s_ramp = 0.1  # 匝道参考线采样步长
    ramp_s = 10

    # 获取匝道起点和终点
    ramp_start_point, ramp_end_point = generate_ramp_points(main_road_start_point, main_road_end_point, s=ramp_s,
                                                            angle=ramp_angle, ramp_length=ramp_length)
    print(ramp_start_point, ramp_end_point)

    # 生成匝道的参考线和车道信息（初始）
    ramp_refline_points, ramp_length = generate_ramp_reference_line(ramp_start_point, ramp_end_point, delta_s_ramp)
    ramp_lane_lines = generate_ramp_lane_lines(ramp_refline_points, delta_s_ramp, default_lane_count,
                                               default_lane_count, main_road_lanes)
    ramp_boundary_lines = generate_ramp_boundary_lines(ramp_lane_lines, ramp_refline_points, default_lane_count,
                                                       default_lane_count)

    main_road_lane_lines = generate_lane_lines(main_road_refline_points, default_lane_count, default_lane_count,
                                               delta_s_main_road, main_road_lanes)
    main_road_boundary_lines = generate_boundary_lines(main_road_lane_lines, main_road_refline_points,
                                                       default_lane_count, default_lane_count)
    
    # 计算匝道边界线与主路边界线的交点
    intersection_points_boundary = calculate_intersection_points(ramp_boundary_lines, main_road_boundary_lines)
    intersection_points_lane = calculate_intersection_points_lane(ramp_lane_lines, main_road_boundary_lines)
    print("Intersection Points:", intersection_points_boundary)

    # 使用 remove_points_at_intersection 函数去除匝道边界线上与主路边界线同侧的点，并更新匝道的边界线
    remove_points_at_intersection(ramp_boundary_lines, intersection_points_boundary, main_road_boundary_lines)
    # 使用 remove_points_at_intersection_lane 函数去除匝道车道线上与主路边界线同侧的点，并更新匝道的车道线
    remove_points_at_intersection_lane(ramp_lane_lines, intersection_points_lane, main_road_boundary_lines)
    # 使用 remove_points_between_boundaries 函数去除主路边界线上交点之间的点
    remove_points_between_boundaries(main_road_boundary_lines, intersection_points_boundary)
    print(main_road_boundary_lines['right'])

    # 调用可视化函数
    visualize_road(main_road_refline_points, ramp_refline_points, main_road_lane_lines, ramp_lane_lines,
                   main_road_boundary_lines, ramp_boundary_lines)
    
# 生成主路和匝道示例
generate_road_example()
