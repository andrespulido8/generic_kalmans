#!/usr/bin/env python3
import numpy as np
from scipy.spatial import distance
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from raite_msgs.msg import PointArray 
from .kf import KalmanFilter


class TrackingNode(Node):
    def __init__(self):
        super().__init__("tracking_node")
        self.init_finished = False

        # Dictionary to store Kalman Filters for multiple targets
        self.kalman_filters = {}

        # KF initialization 
        self.measurement_noise = 0.01
        self.max_measurement_noise = 0.2
        self.measurement_covariance = np.diag([self.measurement_noise]*2)
        self.process_covariance = np.diag([0.00005, 0.00005, 0.0004, 0.0004])
        self.initial_cov = np.diag([0.1, 0.1, 0.02, 0.02])

        self.last_time = None
        self.last_target_id = -1
        self.estimate_time_cutoff = 5.0  # seconds
        self.lock = False  # prevent changing memory while publish estimate is running

        # ROS stuff
        self.get_logger().info("Initializing tracking_node")

        self.overhead_sub = self.create_subscription(
            PointArray, "/overhead_camera/homography_pose", self.overhead_camera_cb, 1
        )
        self.side_sub = self.create_subscription(
            PointArray, "/side_camera/homography_pose", self.side_camera_cb, 1
        )
        self.side2_sub = self.create_subscription(
            PointArray, "/side_camera2/homography_pose", self.side_camera2_cb, 1
        )
        self.rate = self.declare_parameter("rate_hz", 20.0).value
        # Publisher to publish estimates for all targets
        self.estimate_pub = self.create_publisher(PoseArray, "/estimate_pub", 10)
        self.timer = self.create_timer(1.0 / self.rate, self.publish_estimates)

    def overhead_camera_cb(self, msg):
        if bool(msg.points) is False:
            return
        measurements = [
                np.array([[point.point.x], [point.point.y]]) for point in msg.points
        ]
        if not self.lock:
            for measurement in measurements:
                self.create_or_update_filter(*self.assign_measurement(measurement))

    def side_camera_cb(self,msg):
        if bool(msg.points) is False:
            return
        measurements = [
                np.array([[point.point.x], [point.point.y]]) for point in msg.points
        ]
        if not self.lock:
            for measurement in measurements:
                self.create_or_update_filter(*self.assign_measurement(measurement))

    def side_camera2_cb(self,msg):
        if bool(msg.points) is False:
            return
        measurements = [
                np.array([[point.point.x], [point.point.y]]) for point in msg.points
        ]
        if not self.lock:
            for measurement in measurements:
                self.create_or_update_filter(*self.assign_measurement(measurement))
    
    def assign_measurement(self, measurement):
        """Assigns the measurement and measurement covariance.
           The assignment is based on the min mahalanobis distance between the measurement 
           and the current state of the Kalman Filter.
        Args:
            measurement: The message received from the camera.
        Returns:
            target_id The ID of the target.
            measurement: The measurement for the target.
            measurement_covariance: The measurement covariance for the target.
            """
        measurement_covariance = self.inv_proportional_noise(1.0)   # TODO: get confidence from detection model

        # Data association
        # use mahalanobis distance to determine which estimator this measurement belongs to
        min_distance = np.inf
        for curr_obj_id, kf in self.kalman_filters.items():
            mahalanobis_distance = distance.mahalanobis(measurement, kf.X[:2], np.linalg.inv(kf.P[:2, :2]))
            if mahalanobis_distance < min_distance:
                min_distance = np.copy(mahalanobis_distance)
                target_id = int(np.copy(curr_obj_id))

        # track creation 
        if min_distance > 4.5:  # approx 1.5 m away with cov 0.1
            if self.last_target_id > 100:
                self.last_target_id = -1  # reset target id
            self.last_target_id += 1
            target_id = self.last_target_id 

        return (target_id, measurement, measurement_covariance)

    def create_or_update_filter(self, target_id, measurement, measurement_covariance):
        """Creates a new Kalman Filter for a target if it does not exist, or updates the existing filter.
        Args:
            target_id: The ID of the target.
            measurement: The measurement for the target.
        """
        # TODO: check velocity of targets to make sure they do not exceed reality (delete their track)
        # TODO: check if only one camera is getting measurements (posisble fake)
        # TODO: check positions outside some bounds and flag as invalid
        # TODO: try different motion models if they are needed when occluded

        if target_id not in self.kalman_filters:
            # Initialize the Kalman Filter for the new target
            initial_state = np.array([measurement[0], measurement[1], 0.1, 0.1])
            self.kalman_filters[target_id] = KalmanFilter(
                self.initial_cov,
                self.process_covariance,
                self.measurement_covariance,
                initial_state,
            )
        else:
            # Update the Kalman Filter with the new measurement
            self.kalman_filters[target_id].update(measurement, measurement_covariance)
            self.kalman_filters[target_id].last_update_time = self.get_clock().now().seconds_nanoseconds()[0]

    def inv_proportional_noise(self, confidence):
        """ Returns a measurement covariance matrix that is inversely proportional to the confidence level.
            The mapping is linear with min_confidence confidence being the maximum measurement noise and 1 confidence being the
            ideal measurement noise measurement_noise.
        Args:
            confidence: The confidence level between min_confidence and 1."""
        min_confidence = 0.1  # from learned detection model
        m = ((self.measurement_noise - self.max_measurement_noise)/(1 - min_confidence))
        return np.diag([m*confidence + self.measurement_noise - m]*2)

    def publish_estimates(self):
        """Publishes the estimates for all targets."""
        self.lock = True
        t = self.get_clock().now().seconds_nanoseconds()[0]
        dt = t - self.last_time if self.last_time else 1 / self.rate

        estimates_to_delete = []
        #print("kfs: ", self.kalman_filters.keys())
        if self.kalman_filters:
            msg = PoseArray()
            msg.header.frame_id = "world"
            for target_id, kf in self.kalman_filters.items():
                kf.predict(1.0 / self.rate)  # Assuming a fixed time step for prediction
                est_msg = Pose()
                est_msg.pose.position.x = kf.X[0]
                est_msg.pose.position.y = kf.X[1]
                est_msg.pose.position.z = 0.  # Assuming 2D tracking for now
                yaw = np.arctan2(kf.X[3], kf.X[2])
                est_msg.pose.orientation.x = 0.
                est_msg.pose.orientation.y = 0.
                est_msg.pose.orientation.z = np.sin(yaw / 2)
                est_msg.pose.orientation.w = np.cos(yaw / 2)
                #cov = np.zeros(36)
                #cov[0] = kf.P[0,0]
                #cov[7] = kf.P[1,1]
                #est_msg.pose.covariance = cov
                msg.poses.append(est_msg)

                # TODO: publish number of target ids

                # estimate deletion
                if kf.last_update_time is not None:
                    #print("last update was: ", t - kf.last_update_time, "sec ago, for id: ", target_id)
                    if (t - kf.last_update_time) > self.estimate_time_cutoff:
                        estimates_to_delete.append(target_id)
                        #print("deleting: ", target_id)

            self.estimate_pub.publish(est_msg)
            for t_id in estimates_to_delete:
                del self.kalman_filters[t_id]
        
        self.last_time = t
        self.lock = False


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()

    try:
        rclpy.spin(tracking_node)
    except KeyboardInterrupt:
        tracking_node.get_logger().info("Shutting down tracking_node")
    finally:
        tracking_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
