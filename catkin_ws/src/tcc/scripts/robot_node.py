#!/usr/bin/env python3


from turtle import done
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
from std_srvs.srv import Empty
import itertools
from dqn import Agent
import math
from math import *

import numpy as np
import tf





class Robot_ros:
	def __init__(self):
		rospy.init_node('robot')
		self.sub = rospy.Subscriber('/odom',Odometry, self.get_pos)
		self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)
		self.rate = rospy.Rate(10)
		self.vel = Twist()

		self.posi_x = 0
		self.posi_y = 0
		self.rang = 0
		self.theta = 0
		self.MAX_LIDAR_DISTANCE = 5
		self.distance = []
		self.angle = []
		self.actions_ = []

	def get_pos(self, data):
		self.posi_x = data.pose.pose.position.x
		self.posi_y = data.pose.pose.position.y
		(roll, pitch, yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
		self.theta = yaw

	def sensor_read(self,msgScan):
		self.distance = []	
		self.rang = msgScan.ranges
		
		for i in range(len(msgScan.ranges)):
			if (msgScan.ranges[i] > self.MAX_LIDAR_DISTANCE ):
				self.distance.append(self.MAX_LIDAR_DISTANCE)
			else:
				self.distance.append(msgScan.ranges[i])
			
		distances = np.array(self.distance,dtype=np.float32)
		
		return  distances

	def create_action_space(self):

		action_space = np.array([0,1,2,3])
		return action_space

	def take_action(self, a):
		if a == 0:
			self.vel.linear.x = 2
			self.vel.linear.y = 0
			self.vel.linear.z = 0	
			self.vel.angular.x = 0
			self.vel.angular.y = 0
			self.vel.angular.z = 0

		if a == 1:
			self.vel.linear.x = 1.2
			self.vel.linear.y = 0
			self.vel.linear.z = 0	
			self.vel.angular.x = 0
			self.vel.angular.y = 0
			self.vel.angular.z = 0.6

		if a == 2:
			self.vel.linear.x = 1.2
			self.vel.linear.y = 0
			self.vel.linear.z = 0	
			self.vel.angular.x = 0
			self.vel.angular.y = 0
			self.vel.angular.z = -0.6

		if a == 3:
			self.vel.linear.x = 0.4
			self.vel.linear.y = 0
			self.vel.linear.z = 0	
			self.vel.angular.x = 0
			self.vel.angular.y = 0
			self.vel.angular.z = 0			


		self.pub.publish(self.vel)
		self.rate.sleep()


	def checkCrash(self,lidar):
		var = 0
		if min(lidar)<0.05:
			var = 1
		return var

	def get_reward(self, action, prev_action, lidar, prev_lidar, crash):

		self.actions_.append(action)

		if crash ==1:
			reward = -100
		else:
			if action==0:
				r_action = 0.2
			else:
				r_action = -0.1

			if min(prev_lidar)<min(lidar):
				r_obs = 0.2
			else:
				r_obs = -0.2

			if ( prev_action == 1 and action == 2 ) or ( prev_action == 2 and action == 1 ):
				r_change = -0.8
			else:
				r_change = 0.0
			
			if (sum(self.actions_) / float(len(self.actions_)))==3:
				r_stop = -10
			else:
				r_stop = 0.0

			
			reward = r_action + r_obs + r_change + r_stop

			if len(self.actions_)==5:
				self.actions_ = []
				

		return reward


	def stop_robot(self):
		self.vel = Twist()
		self.vel.linear.x = 0
		self.vel.linear.y = 0
		self.vel.linear.z = 0	
		self.vel.angular.x = 0
		self.vel.angular.y = 0
		self.vel.angular.z = 0
		self.pub.publish(self.vel)
		self.rate.sleep()
	
	def back_gr(self):
		rospy.wait_for_service('/reset_positions')
		self.clear_bg = rospy.ServiceProxy('/reset_positions', Empty)
		self.clear_bg()



epochs = 500
num_max_steps = 3000

robot = Robot_ros()
action_space = robot.create_action_space()

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, 
            n_actions=len(action_space), input_dims=270, lr=0.003)


update_target_ = 10

robot.back_gr()


scores =  []
while not rospy.is_shutdown():
	
	for e in range(epochs):
		state = robot.sensor_read(rospy.wait_for_message('/base_scan', LaserScan))
		lidar = state 
		prev_action =  agent.get_an_action(state)
		prev_lidar  = lidar
		crash = 0
		score = 0 
		steps = 0
		done = False

		while not done:
			
			state = robot.sensor_read(rospy.wait_for_message('/base_scan', LaserScan))
			action =  agent.get_an_action(state)
			
			robot.take_action(action)
			new_state = robot.sensor_read(rospy.wait_for_message('/base_scan', LaserScan))
			
			lidar = new_state
			crash = robot.checkCrash(lidar)

			if crash==1 or steps==num_max_steps: 
				done = True
				robot.back_gr()
			else:
				done = False	
			
			reward = robot.get_reward(action, prev_action, lidar, prev_lidar, crash)

			score+=reward
			agent.save_transitions(state, action, reward, new_state, done)
			agent.learn()

			state = new_state
			prev_lidar = lidar
			prev_action = action

			steps+=1

		scores.append(score)
		avg_score = np.mean(scores[-100:])
		print('epoch: ',e, ' score %f', score, 'avg_score %f', avg_score)

		if e%10 == 0:
			agent.save()

		if e%update_target_==0:
			agent.update_target()

		
		

					


			
	