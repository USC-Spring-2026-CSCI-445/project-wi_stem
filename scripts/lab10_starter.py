#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi,sin,cos
from time import sleep, time
import queue
import json

import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = []
        self.t_prev = 0
        self.u_min = u_min
        self.u_max = u_max

    def control(self, err, t):
        dt = t - self.t_prev
        if dt < 1e-6 :
            dt = 1e-6
        self.err_hist.append(err)
        self.err_int += err
        if len(self.err_hist) > self.kS:
            self.err_int -= self.err_hist.pop(0)
        self.err_dif = err - self.err_prev
        u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
        self.err_prev = err
        self.t_prev = t
        return max(self.u_min, min(u, self.u_max))


class Node:
    def __init__(self, position: POSITION_TYPE, parent: "Node"):
        self.position = position
        self.neighbors = []
        self.parent = parent

    def distance_to(self, other_node: "Node") -> float:
        return np.linalg.norm(self.position - other_node.position)

    def to_dict(self) -> Dict:
        return {"x": self.position[0], "y": self.position[1]}

    def __str__(self) -> str:
        return (
            f"Node<pos: {round(self.position[0], 4)}, {round(self.position[1], 4)}, #neighbors: {len(self.neighbors)}>"
        )


class RrtPlanner:

    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb
        self.graph_publisher = rospy.Publisher("/rrt_graph", MarkerArray, queue_size=10)
        self.plan_visualization_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.delta = 0.25
        self.obstacle_padding = 0.15
        self.goal_threshold = GOAL_THRESHOLD

    def visualize_plan(self, path: List[Dict]):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.scale = Vector3(0.075, 0.075, 0.1)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
            marker_array.markers.append(marker)
        self.plan_visualization_pub.publish(marker_array)

    def visualize_graph(self, graph: List[Node]):
        marker_array = MarkerArray()
        for i, node in enumerate(graph):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale = Vector3(0.05, 0.05, 0.05)
            marker.pose.position = Point(node.position[0], node.position[1], 0.01)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
            marker_array.markers.append(marker)
        self.graph_publisher.publish(marker_array)

    def _randomly_sample_q(self) -> Node:
        # Choose uniform randomly sampled points
        ######### Your code starts here #########
        x_min, x_max, y_min, y_max = self.map_aabb
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return Node(np.array([x, y]), None)

        ######### Your code ends here #########

    def _nearest_vertex(self, graph: List[Node], q: Node) -> Node:
        # Determine vertex nearest to sampled point
        ######### Your code starts here #########
        nearest = min(graph, key=lambda node: node.distance_to(q))
        return nearest
        ######### Your code ends here #########

    def _is_in_collision(self, q_rand: Node):
        x = q_rand.position[0]
        y = q_rand.position[1]
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            x_min -= self.obstacle_padding
            y_min -= self.obstacle_padding
            x_max += self.obstacle_padding
            y_max += self.obstacle_padding
            if (x_min < x and x < x_max) and (y_min < y and y < y_max):
                return True
        return False

    def _extend(self, graph: List[Node], q_rand: Node):

        # Check if sampled point is in collision and add to tree if not
        ######### Your code starts here #########
        q_near = self._nearest_vertex(graph, q_rand)
        direction = q_rand.position - q_near.position
        dist = np.linalg.norm(direction)
        
        if dist == 0:
            return None
        # Steer only up to self.delta
        if dist > self.delta:
            q_new_pos = q_near.position + (direction / dist) * self.delta
        else:
            q_new_pos = q_rand.position.copy()
        q_new = Node(q_new_pos, q_near)
        
        # Check the whole edge from q_near to q_new, not just endpoint
        edge_vec = q_new.position - q_near.position
        edge_len = np.linalg.norm(edge_vec)
        
        n_checks = max(2, int(np.ceil(edge_len / 0.02)))
        for alpha in np.linspace(0.0, 1.0, n_checks):
            interp_pos = q_near.position + alpha * edge_vec
            interp_node = Node(interp_pos, None)
            if self._is_in_collision(interp_node):
                return None
        q_near.neighbors.append(q_new)
        graph.append(q_new)
        return q_new
        ######### Your code ends here #########

    def generate_plan(self, start: POSITION_TYPE, goal: POSITION_TYPE) -> Tuple[List[POSITION_TYPE], List[Node]]:
        """Public facing API for generating a plan. Returns the plan and the graph.

        Return format:
            plan:
            [
                {"x": start["x"], "y": start["y"]},
                {"x": ...,      "y": ...},
                            ...
                {"x": goal["x"],  "y": goal["y"]},
            ]
            graph:
                [
                    Node<pos: x1, y1, #neighbors: n_1>,
                    ...
                    Node<pos: x_n, y_n, #neighbors: z>,
                ]
        """
        graph = [Node(np.array([start["x"], start["y"]]), None)]
        goal_node = Node(np.array([goal["x"], goal["y"]]), None)
        plan = []

        # Find path from start to goal location through tree
        ######### Your code starts here #########
        max_iterations = 5000
        reached_goal_node = None

        for _ in range(max_iterations):
            q_rand = self._randomly_sample_q()
            q_new = self._extend(graph, q_rand)

            if q_new is None:
                continue

            # If close enough to goal, try to connect directly to the goal
            if q_new.distance_to(goal_node) <= self.goal_threshold:
                edge_vec = goal_node.position - q_new.position
                edge_len = np.linalg.norm(edge_vec)

                collision_free_to_goal = True
                n_checks = max(2, int(np.ceil(edge_len / 0.02)))
                for alpha in np.linspace(0.0, 1.0, n_checks):
                    interp_pos = q_new.position + alpha * edge_vec
                    interp_node = Node(interp_pos, None)
                    if self._is_in_collision(interp_node):
                        collision_free_to_goal = False
                        break

                if collision_free_to_goal:
                    reached_goal_node = Node(goal_node.position.copy(), q_new)
                    q_new.neighbors.append(reached_goal_node)
                    graph.append(reached_goal_node)
                    break

        if reached_goal_node is None:
            rospy.logwarn("RRT failed to find a path within the iteration limit.")
            return plan, graph

        # Backtrack from goal to start using parent pointers
        curr = reached_goal_node
        reverse_plan = []
        while curr is not None:
            reverse_plan.append(curr.to_dict())
            curr = curr.parent

        plan = reverse_plan[::-1]

        ######### Your code ends here #########
        return plan, graph


# Protip: copy the ObstacleFreeWaypointController class from lab5.py here
######### Your code starts here #########
# Class for controlling the robot to reach a goal position
class ObstacleFreeWaypointController:
    def __init__(self, waypoints: List[Dict]):
        # rospy.init_node("waypoint_follower", anonymous=True)
        self.waypoints = waypoints
        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        # publish_waypoints(self.waypoints, self.waypoint_pub)

        self.current_position = None

        # define linear and angular PID controllers here
        ######### Your code starts here #########
        self.linear_pid = PIDController(
                kP=0.8,
                kI=0.0,
                kD=0.03,
                kS=0.0,
                u_min=0.0,
                u_max=0.25
                )
        self.angular_pid = PIDController(
                kP=1.2,
                kI=0.0,
                kD=0.05,
                kS=0.0,
                u_min=-0.8,
                u_max=0.8
                )

        ######### Your code ends here #########

    def odom_callback(self, msg):
        # Extracting current position from Odometry message
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None

        # Calculate error in position and orientation
        ######### Your code starts here #########
        dx = goal_position["x"] - self.current_position["x"]
        dy = goal_position["y"] - self.current_position["y"]
        distance_error = sqrt(dx*dx + dy*dy)
        target_theta = atan2(dy, dx)

        angle_error = target_theta - self.current_position["theta"]
        angle_error = angle_to_0_to_2pi(atan2(sin(angle_error), cos(angle_error)))
        # angle_error = atan2(sin(angle_error), cos(angle_error))
        
        ######### Your code ends here #########

        return distance_error, angle_error

    def control_robot(self):
        rate = rospy.Rate(20)  # 20 Hz
        ctrl_msg = Twist()

        current_waypoint_idx = 0

        while not rospy.is_shutdown():

            # Travel through waypoints one at a time, checking if robot is close enough
            ######### Your code starts here #########
            if self.current_position is None:
                rate.sleep()
                continue

            # If we've already reached all waypoints, stop and just keep publishing zero
            if current_waypoint_idx >= len(self.waypoints):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.robot_ctrl_pub.publish(ctrl_msg)
                rate.sleep()
                continue

            # Current target waypoint
            goal = self.waypoints[current_waypoint_idx]
            error = self.calculate_error(goal)
            
            if error is None:
                rate.sleep()
                continue

            distance_error, angle_error = error

            # If we are close enough to this waypoint, move to the next one
            if distance_error < 0.1:
                rospy.loginfo(
                    f"Reached waypoint {current_waypoint_idx}: {goal}")
                current_waypoint_idx += 1
                continue

            t = rospy.get_time()
            w = self.angular_pid.control(angle_error, t)
            # v = self.linear_pid.control(distance_error, t)
            
            if abs(angle_error) > 0.7:
                v = 0.0 
            else:
                v = self.linear_pid.control(distance_error,t)
            ctrl_msg.linear.x = v
            ctrl_msg.angular.z = w

            self.robot_ctrl_pub.publish(ctrl_msg)

            
            ######### Your code ends here #########
            rate.sleep()

######### Your code ends here #########


""" Example usage

rosrun development lab10.py --map_filepath src/csci445l/scripts/lab10_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        goal_position = map_["goal_position"]
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]
        start_position = {"x": 0.0, "y": 0.0}

    rospy.init_node("rrt_planner")
    planner = RrtPlanner(obstacles, map_aabb)
    plan, graph = planner.generate_plan(start_position, goal_position)
    planner.visualize_plan(plan)
    planner.visualize_graph(graph)
    controller = ObstacleFreeWaypointController(plan)

    try:
        while not rospy.is_shutdown():
            controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")

