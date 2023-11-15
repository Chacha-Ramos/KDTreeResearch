import random
import numpy as np


class Datapoint:
    def __init__(self, coordinate, color, **kwargs):
        assert isinstance(coordinate, list), 'Coordinate must be a list of integers'
        assert isinstance(color, str)
        self.coordinate = coordinate
        self.color = color
        self.kwargs = kwargs


class Node:
    def __init__(self, weight = 0, children = 0, color = None, rect_1 = None, rect_2 = None, dp: Datapoint = None,
                 **kwargs):
        self.left_rectangle: list[int] or None = rect_1
        self.right_rectangle: list[int] or None = rect_2
        self.datapoint: Datapoint or None = dp
        self.left_child: Node or None = None
        self.right_child: Node or None = None
        self.color: str or None = color
        self.weight = weight
        self.children = children
        self.kwargs = kwargs


class KDTree:
    def __init__(self):
        self.root: Node or None = None
        self.dimension: int = 0
        self.level1: dict[str: float] = dict()
        self.colors: dict[str: float] or None = None
        self.twod_root: Node or None = None

    def __calculate_probs(self, weights: list[float]) -> list[float]:
        ai = list()
        for weight in weights:
            ai.append(self.__single_prob(weight))
        return ai

    def __single_prob(self, weight) -> float:
        return random.random() ** (1 / weight)

    def __add_color(self, color: str):
        if color not in self.level1:
            self.level1[color] = self.__single_prob(self.colors[color])

    def __twod_transformation(self, points: list[Datapoint]):
        points.sort(key=lambda p: p.coordinate)
        colors = dict()
        twod_points = list()
        for p in points:
            value = -10
            if p.color in colors:
                value = colors[p.color]
            colors[p.color] = p.coordinate[0]
            twod_point = Datapoint([p.coordinate[0], value], p.color)
            twod_points.append(twod_point)
        return twod_points

    def __can_nodes_break(self, nodes):
        if len(nodes) == 1:
            if nodes[0].datapoint is not None:
                return nodes[0].color
            else:
                n = [nodes[0].left_child, nodes[0].right_child]
                return self.__can_nodes_break(n)
        else:
            weights = [self.colors[i.color] for i in nodes]
            ai = self.__calculate_probs(weights)
            index_max = np.argmax(ai)
            n = [nodes[index_max]]
            return self.__can_nodes_break(n)

    def level2_random_selection(self, rectangle: list, points: list[Datapoint]):
        if self.twod_root is None:
            twod_points = self.__twod_transformation(points)
            self.twod_root = self.__build_tree(twod_points)
        can_nodes = self.__query_canonical(self.twod_root, rectangle)
        if len(can_nodes) == 0:
            return can_nodes, self.twod_root
        return self.__can_nodes_break(can_nodes)

    def level1_random_selection(self, rectangle: list) -> str:
        c_nodes = self.query_canonical(rectangle)
        weights = [self.level1[node.color] for node in c_nodes]
        index_max = np.argmax(weights)
        return c_nodes[index_max].color

    def build_tree(self, points: list[Datapoint], colors: dict[str: float]):
        assert len(points) > 0, 'There must be at least one datapoint to build tree'
        assert len(points[0].coordinate), 'The coordinate must have at least one dimension'
        self.dimension = len(points[0].coordinate)
        self.colors = colors
        self.root = self.__build_tree(points)

    def __build_tree(self, points: list[Datapoint]) -> Node or None:
        if len(points) == 0:
            return None
        if len(points) == 1:
            color_1 = points[0].color
            self.__add_color(color_1)
            return Node(dp=points[0], color=color_1, weight=self.colors[color_1])
        if len(points) == 2:
            points.sort(key=lambda pts: pts.coordinate)

            if len(points[0].coordinate) == 1:
                r_1 = [[points[0].coordinate[0], points[0].coordinate[0]],
                       [points[0].coordinate[0], points[0].coordinate[0]]]
            else:
                r_1 = [points[0].coordinate, points[0].coordinate]
            if len(points[1].coordinate) == 1:
                r_2 = [[points[1].coordinate[0], points[1].coordinate[0]],
                       [points[1].coordinate[0], points[1].coordinate[0]]]
            else:
                r_2 = [points[1].coordinate, points[1].coordinate]

            color_1 = points[0].color
            color_2 = points[1].color
            self.__add_color(color_1)
            self.__add_color(color_2)
            if self.level1[color_1] >= self.level1[color_2]:
                root_color = color_1
            else:
                root_color = color_2

            root = Node(rect_1=r_1, rect_2=r_2, color=root_color, children=2,
                        weight=self.colors[color_1] + self.colors[color_2])
            root.left_child = Node(dp=points[0], color=color_1, weight=self.colors[color_1])
            root.right_child = Node(dp=points[1], color=color_2, weight=self.colors[color_2])
            return root

        coordinates = np.array([p.coordinate for p in points])
        variances = np.var(coordinates, axis=0)
        index_max_variance = np.argmax(variances)
        points.sort(key=lambda pts: pts.coordinate[index_max_variance])
        median_index = len(points) // 2
        root = Node(children=len(points))
        p1 = points[:median_index]
        p2 = points[median_index:]
        p1.sort(key=lambda pts: pts.coordinate)
        p2.sort(key=lambda pts: pts.coordinate)
        if p1[0].coordinate <= p2[0].coordinate:
            root.left_child = self.__build_tree(p1)
            root.right_child = self.__build_tree(p2)
        else:
            root.left_child = self.__build_tree(p2)
            root.right_child = self.__build_tree(p1)

        if root.right_child.datapoint is not None:
            r_1 = root.right_child.datapoint.coordinate
            if len(r_1) == 1:
                root.right_rectangle = [[r_1[0], r_1[0]], [r_1[0], r_1[0]]]
            else:
                root.right_rectangle = [r_1, r_1]
        else:
            right_child = root.right_child
            min_l = np.min(right_child.left_rectangle + right_child.right_rectangle, axis=0)
            max_r = np.max(right_child.left_rectangle + right_child.right_rectangle, axis=0)

            r_1 = [min_l[0], min_l[1]]
            r_2 = [max_r[0], max_r[1]]

            root.right_rectangle = [r_1, r_2]

        if root.left_child.datapoint is not None:
            r_1 = root.left_child.datapoint.coordinate
            if len(r_1) == 1:
                root.left_rectangle = [[r_1[0], r_1[0]], [r_1[0], r_1[0]]]
            else:
                root.left_rectangle = [r_1, r_1]
        else:
            left_child = root.left_child
            min_l = np.min(left_child.left_rectangle + left_child.right_rectangle, axis=0)
            max_r = np.max(left_child.left_rectangle + left_child.right_rectangle, axis=0)

            r_1 = [min_l[0], min_l[1]]
            r_2 = [max_r[0], max_r[1]]

            root.left_rectangle = [r_1, r_2]

        color_1 = root.left_child.color
        color_2 = root.right_child.color
        if self.level1[color_1] >= self.level1[color_2]:
            root_color = color_1
        else:
            root_color = color_2

        root.weight = root.left_child.weight + root.right_child.weight
        root.color = root_color
        return root

    def query_canonical(self, rectangle: list):
        assert len(rectangle) == 2, f'There must be two coordinates to draw a query rectangle'
        assert isinstance(rectangle[0], list) and isinstance(rectangle[0],
                                                             list), 'Each coordinate must be in the format of a list'
        assert len(rectangle[0]) == self.dimension and len(
            rectangle[1]) == self.dimension, f'Expected dimension of rectangle to be {self.dimension}'
        assert rectangle[0] <= rectangle[1], 'Invalid query rectangle'
        return self.__query_canonical(self.root, rectangle)

    def __rectangles_intersect(self, root_rectangle, query_rectangle):
        for dim in range(len(query_rectangle[0])):
            if root_rectangle[0][dim] > query_rectangle[1][dim] or query_rectangle[0][dim] > root_rectangle[1][dim]:
                return False
        return True

    def __query_canonical(self, root: Node or None, query_rect: list) -> list[Node]:
        canonical_nodes: list[Node] = list()
        if root is None:
            return canonical_nodes

        if root.datapoint is not None:
            if np.all(np.array(query_rect[0]) <= np.array(root.datapoint.coordinate)) and np.all(
                    np.array(root.datapoint.coordinate) <= np.array(query_rect[1])):
                canonical_nodes.append(root)
            return canonical_nodes

        if np.all(np.array(query_rect[0]) <= np.array(root.left_rectangle[0])) and np.all(
                np.array(query_rect[1]) >= np.array(root.right_rectangle[1])):
            canonical_nodes.append(root)
            return canonical_nodes

        if self.__rectangles_intersect(root.left_rectangle, query_rect):
            canonical_nodes.extend(self.__query_canonical(root.left_child, query_rect))

        if self.__rectangles_intersect(root.right_rectangle, query_rect):
            canonical_nodes.extend(self.__query_canonical(root.right_child, query_rect))

        return canonical_nodes
