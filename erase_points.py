import numpy as np 
from autolab_core import RigidTransform
from frankapy import FrankaArm
import traceback

# We extend functionality from the base franka class
class FrankaSteinArm(FrankaArm):
	def __init__(self):
		# Start default functionality of the franka arm
		super().__init__()

		# We need to save some special states and info
		# Pose where the robot grasps the eraser
		self.default_gripper_orientation = np.array(
				[[1.,  0.,  0.],
				[ 0., -1.,  0.],
				[ 0.,  0., -1.]]) # CHANGE IF NEEDED
		self.eraser_position = np.array([0.51414798, 0.36158461, 0.01890064]) # CHANGE IF NEEDED
		
		self.primer_delta = np.array([0., 0., 0.2]) # Safety so as not to collide with eraser

		self.eraser_grasping_pose = RigidTransform(
			rotation = self.default_gripper_orientation,
			translation=self.eraser_position,
			from_frame="franka_tool", to_frame="world")

		self.eraser_grasping_primed = RigidTransform(
			rotation = self.default_gripper_orientation,
			translation=self.eraser_position + self.primer_delta,
			from_frame="franka_tool", to_frame="world")

		# Useful for surface sliding
		self.surface_level = 0.01

	def catch_eraser(self):
		print("Getting eraser")
		print("Moving to eraser prime")
		self.goto_pose(self.eraser_grasping_primed, use_impedance=False)
		print("Moving to grasp eraser")
		self.goto_pose(self.eraser_grasping_pose, use_impedance=False)
		print("Grasping eraser")
		self.close_gripper() # Will close until it hits the object. Pretty tight grasp
		print("Lifting eraser")
		self.goto_pose(self.eraser_grasping_primed, use_impedance=False)

	def return_eraser(self):
		print("Returning eraser")
		print("Moving to eraser prime")
		self.goto_pose(self.eraser_grasping_primed, use_impedance=False)
		print("Moving to place eraser")
		self.goto_pose(self.eraser_grasping_pose, use_impedance=False)
		print("Letting go of eraser")
		self.open_gripper() # Will close until it hits the object. Pretty tight grasp
		print("Lifting hand")
		self.goto_pose(self.eraser_grasping_primed, use_impedance=False)
		print("Sending home")
		self.reset_joints()


	def erase(self, trajectory):
		"""
		Erases surface between 2 points
		trajectory: list of points (x, y) indicating a trajectory to clean
		degrades with distance?...
		TODO: Integrate with Ishir's points
		"""

		if (len(trajectory) >= 2):
			# Prime for movement in first point
			print("Priming for trajectory")
			p1x, p1y = trajectory[0]
			traj_primer = RigidTransform(
				rotation = self.default_gripper_orientation,
				translation=np.array([p1x, p1y, self.surface_level]) + self.primer_delta,
				from_frame="franka_tool", to_frame="world")
			self.goto_pose(traj_primer, use_impedance=False)

			# Move to first pt
			print("Setting on trajectory")
			traj_start = RigidTransform(
				rotation = self.default_gripper_orientation,
				translation=np.array([p1x, p1y, self.surface_level]),
				from_frame="franka_tool", to_frame="world")
			self.goto_pose(traj_start, use_impedance=False)

			# Go to point
			n_pts = len(trajectory)
			for i in range(n_pts-1):
				curr_pt = trajectory[i]
				next_pt = trajectory[i+1]

				# Find orientation between points
				x_hat = np.append(next_pt - curr_pt, 0)
				x_hat = x_hat/np.linalg.norm(x_hat)
				# Arm can't turn more than 90 deg without glitching
				if x_hat@np.array([1, 0, 0]) < 0:
					x_hat = -x_hat

				z_hat = np.array([0, 0, -1])

				y_hat = np.cross(z_hat, x_hat)

				rotation_matrix = np.transpose(np.stack([x_hat, y_hat, z_hat], axis=0))

				print("Reaching orientation")
				traj_start = RigidTransform(
				rotation = rotation_matrix,
				translation=np.array([curr_pt[0], curr_pt[1], self.surface_level]),
				from_frame="franka_tool", to_frame="world")
				self.goto_pose(traj_start, use_impedance=False)

				print("Erasing segment")
				traj_end = RigidTransform(
				rotation = rotation_matrix,
				translation=np.array([next_pt[0], next_pt[1], self.surface_level]),
				from_frame="franka_tool", to_frame="world")
				self.goto_pose(traj_end, use_impedance=False)

			# Lift eraser after trajcetory is done
			print("Lifting eraser")
			traj_lift = RigidTransform(
			rotation = self.default_gripper_orientation,
			translation=np.array([trajectory[-1][0], trajectory[-1][1], self.surface_level]) + self.primer_delta,
			from_frame="franka_tool", to_frame="world")
			self.goto_pose(traj_lift, use_impedance=False)



if __name__ == "__main__":
	try:
		fa = FrankaSteinArm()
		print("Resetting joints")
		fa.reset_joints()
		fa.catch_eraser()

		# Trajectory points
		traj1 = np.array([
			[0.45, -0.35],
			[0.51, -0.24],
			[0.60, -0.21]])

		traj2 = np.array([
			[0.62, -0.07],
			[0.49, -0.01],
			[0.44, 0.11]])

		trajectories = [traj1, traj2]
		for traj in trajectories:
			fa.erase(traj)


		fa.return_eraser()
	except:
		print("AN ERROR OCCURRED")
		traceback.print_exc()
		print("Initiating reset protocol")
		fa.return_eraser()