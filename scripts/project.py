#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import (
    Map,
    ParticleFilter,
    angle_to_neg_pi_to_pi,
)
from lab10_starter import (
    RrtPlanner,
    PIDController as WaypointPID,
    GOAL_THRESHOLD,
)


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(
        self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]
    ):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (
            not rospy.is_shutdown()
        ):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (
            not rospy.is_shutdown()
        ):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (
            not rospy.is_shutdown()
        ):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 5 beams (-15°, 0°, +15°, 90°, -90°) to heavily stabilize PF tracking!
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        offset15 = int(15.0 / (angle_increment * 180.0 / math.pi))
        offset90 = int(90.0 / (angle_increment * 180.0 / math.pi))

        indices = [num_ranges - offset15, 0, offset15, num_ranges - offset90, offset90]

        measurements = []
        for idx in indices:
            idx = idx % num_ranges
            z = ranges[idx]
            if z == inf or np.isinf(z) or math.isnan(z):
                z = getattr(self.laserscan, "range_max", 10.0)
            angle = angle_min + idx * angle_increment
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """

        ######### Your code starts here #########
        rospy.loginfo("Starting Phase 1: PF Localization...")

        confidence_pos_thresh = 0.15
        confidence_theta_thresh = 0.35

        import random

        for step in range(max_steps):
            if rospy.is_shutdown():
                break

            self.take_measurements()
            self._pf.resample()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            particles = self._pf._particles
            if len(particles) > 0:
                xs = [p.x for p in particles]
                ys = [p.y for p in particles]
                thetas = [p.theta for p in particles]

                std_x = np.std(xs)
                std_y = np.std(ys)

                mean_sin = np.mean([math.sin(t) for t in thetas])
                mean_cos = np.mean([math.cos(t) for t in thetas])
                R = math.sqrt(mean_sin**2 + mean_cos**2)
                theta_spread = 1.0 - R

                if (
                    std_x < confidence_pos_thresh
                    and std_y < confidence_pos_thresh
                    and theta_spread < confidence_theta_thresh
                ):
                    rospy.loginfo(f"PF converged legitimately at step {step}")
                    break

            scan = self.laserscan
            if scan is None or len(scan.ranges) == 0:
                self.rate.sleep()
                continue

            front_indices = list(range(0, 15)) + list(
                range(len(scan.ranges) - 15, len(scan.ranges))
            )
            front_ranges = []
            for i in front_indices:
                z = scan.ranges[i]
                if z != float("inf") and not math.isnan(z):
                    front_ranges.append(z)

            min_front = min(front_ranges) if len(front_ranges) > 0 else float("inf")

            if min_front > 0.35:
                self.move_forward(0.2)
            else:
                self.move_forward(-0.05)  # Backup clearance
                turn_angle = random.choice([math.pi / 2, -math.pi / 2])
                self.rotate_in_place(turn_angle)

            self.rate.sleep()
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        rospy.loginfo("Starting Phase 2: Planning with RRT...")

        est_x, est_y, est_theta = self._pf.get_estimate()
        start = {"x": est_x, "y": est_y, "theta": est_theta}

        self.plan, graph = self._planner.generate_plan(start, self.goal_position)

        self._planner.visualize_graph(graph)
        self._planner.visualize_plan(self.plan)

        self.current_wp_idx = 0
        plan_length = len(self.plan) if self.plan else 0
        rospy.loginfo(f"Phase 2 complete. Plan length: {plan_length}")
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########
        rospy.loginfo("Starting Phase 3: Following Plan...")
        plan_list = self.plan
        if not plan_list:
            rospy.logwarn("No plan to follow.")
            return

        ctrl_msg = Twist()
        step_counter = 0

        while not rospy.is_shutdown():
            if self.current_wp_idx >= len(plan_list):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.cmd_pub.publish(ctrl_msg)
                rospy.loginfo("Reached the goal!")
                break

            step_counter += 1
            if step_counter % 5 == 0:
                self.take_measurements()
                self._pf.resample()

            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            goal = plan_list[self.current_wp_idx]

            # We distance-track AND angle-track against the Global Map (Particle Filter) to eliminate drift...
            est_x, est_y, est_theta = self._pf.get_estimate()
            dx = goal["x"] - est_x
            dy = goal["y"] - est_y
            distance_error = math.hypot(dx, dy)
            target_theta = math.atan2(dy, dx)

            # CORRECT angle_error computation (Global Target Angle vs Global PF Robot Angle)
            angle_error = angle_to_neg_pi_to_pi(target_theta - est_theta)

            if distance_error < 0.15:
                rospy.loginfo(f"Reached waypoint {self.current_wp_idx}")
                self.current_wp_idx += 1
                continue

            t = rospy.get_time()
            u_w = self.angular_pid.control(angle_error, t)

            # Exactly Lab 10's 0.7 bound
            if abs(angle_error) > 0.7:
                u_v = 0.0
            else:
                u_v = self.linear_pid.control(distance_error, t)

            ctrl_msg.linear.x = u_v
            ctrl_msg.angular.z = u_w
            self.cmd_pub.publish(ctrl_msg)
            self.rate.sleep()
        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 300
    translation_variance = 0.003
    rotation_variance = 0.03
    measurement_variance = 0.25

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass

