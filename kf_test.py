from raite_tracking_pkg.raite_tracking_pkg.kf import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_ellipse(estimated_position, covariance, color="green"):
    # calculate the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    # calculate the angle of the eigenvector with the largest eigenvalue
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    # calculate the width and height of the ellipse
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])
    # plot the ellipse
    ellipse = plt.matplotlib.patches.Ellipse(estimated_position, width, height, angle=angle * 180 / np.pi, alpha=0.2, color=color)
    plt.gca().add_patch(ellipse)

# Read data from file homography_run.csv with pandas. The first row is the header
data = pd.read_csv("20241015homography_data.csv", header=0)
homography_names = ["overhead camera homography", "side camera homography", "side camera2 homography"]
name_to_confidence = {homography_names[ii]: 0.1+0.2*ii for ii in range(len(homography_names))}

# Normalize time
for name in homography_names + ["leo enu"]:
    data[f"{name} pose time"] = data[f"{name} pose time"].to_numpy() - data[f"{name} pose time"].to_numpy()[0]

# Combine x and y positions for homography and leo poses
leo_pose = np.vstack((data["leo enu pose x"].to_numpy(), data["leo enu pose y"].to_numpy())).T
leo_times = data["leo enu pose time"].to_numpy()
homography_poses = {}
homography_times = {}
for h_name in homography_names:
    homography_poses[h_name] = np.vstack((data[f"{h_name} pose x"].dropna().to_numpy(), data[f"{h_name} pose y"].dropna().to_numpy())).T
    homography_times[h_name] = data[f"{h_name} pose time"].dropna().to_numpy()

# TODO: compute error rms for all measurements

measurement_covariance = np.diag([0.1, 0.1])
process_covariance = np.diag([0.00005, 0.00005, 0.0004, 0.0004])
intial_cov = np.diag([0.1, 0.1, 0.02, 0.02])

initial_state = np.array([homography_poses[homography_names[0]][0, 0], homography_poses[homography_names[0]][0, 1], 0.1, 0.2])
ekfE = KalmanFilter(intial_cov, process_covariance, measurement_covariance, initial_state)

# Initialize lists for storing estimated positions
estimated_positions = [initial_state[:2]]
estimated_covariances = [ekfE.P[:2, :2]]

dt = 0.1
time_range = np.arange(0, leo_times[-1], dt)
for t in time_range:
    #print("time: %.4f" % timeh[ii])

    # Predict the next state
    ekfE.predict(dt)
    # Update Kalman Filter with new measurements
    for h_name in homography_names:
        if not (ekfE.X[0] > -1.4 and ekfE.X[0] < -1.0) and h_name == "side camera2 homography":
            # measurement is the pose at the time that is closest to t
            idx = np.argmin(np.abs(homography_times[h_name] - t))
            measurement = homography_poses[h_name][idx]

            max_measurement_noise = 0.4
            np.diag([max_measurement_noise - name_to_confidence[h_name] * (max_measurement_noise - measurement_covariance[0,0])]*2)
            ekfE.update(measurement, measurement_covariance)
    estimated_positions.append(ekfE.X[:2])
    estimated_covariances.append(ekfE.P[:2, :2])

# Convert lists to numpy arrays for further processing or plotting
estimated_positions = np.array(estimated_positions)
estimated_covariances = np.array(estimated_covariances)

plt.figure()
# plot 3 sigma bounds using the estimated covariance
xsigma = 3 * np.sqrt(estimated_covariances[:-1, 0, 0])
ysigma = 3 * np.sqrt(estimated_covariances[:-1, 1, 1])
plt.fill_between(time_range, estimated_positions[:-1, 0] - xsigma,
                 estimated_positions[:-1, 0] + xsigma, alpha=0.05, color="blue", label="3 sigma x bounds")
plt.fill_between(time_range, estimated_positions[:-1, 1] - ysigma,
                    estimated_positions[:-1, 1] + ysigma, alpha=0.05, color="red", label="3 sigma y bounds")
plt.plot(leo_times, leo_pose[:, 0], ".-", color="blue", label="x true", alpha=0.3)
plt.plot(leo_times, leo_pose[:, 1], ".-", color="red", label="y true", alpha=0.3)
for h_name in homography_names:
    plt.plot(homography_times[h_name], homography_poses[h_name][:, 0], ".", color="blue", label="x measured", alpha=0.1)
    plt.plot(homography_times[h_name], homography_poses[h_name][:, 1], ".", color="red", label="y measured", alpha=0.1)
plt.plot(time_range, estimated_positions[:-1, 0], "*", color="blue", label="x estimated", alpha=0.5)
plt.plot(time_range, estimated_positions[:-1, 1], "*", color="red", label="y estimated", alpha=0.5)
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.legend()
plt.show()

# plot the estimated positions in 2D
plt.figure()
colors = ["blue", "orange", "green"]
ii = 0
for homography_name in homography_names:
    plt.plot(homography_poses[homography_name][:, 0], homography_poses[homography_name][:, 1], ".", color=colors[ii], label=f"{homography_name}")
    ii += 1
plt.plot(leo_pose[:, 0], leo_pose[:, 1], ".-", color="red", label="true")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()

# plot the estimated positions in 2D
plt.figure()
for homography_name in homography_names:
    plt.plot(homography_poses[homography_name][:, 0], homography_poses[homography_name][:, 1], ".", color='blue', label=f"{homography_name}")
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], ".-", color="green", label="estimated")
plt.plot(leo_pose[:, 0], leo_pose[:, 1], ".-", color="red", label="true")
# plot the covariance elipses for every every_n_points points
every_n_points = 10
for ii in range(0, estimated_positions.shape[0], every_n_points):
    draw_ellipse(estimated_positions[ii], estimated_covariances[ii], color="green")
# hline at x=-1.4 and x=-1.0
plt.axvline(x=-1.4, color="red", linestyle="--")
plt.axvline(x=-1.0, color="red", linestyle="--")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()
