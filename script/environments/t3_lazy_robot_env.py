#! usr/bin/env python

'''
    Turtlebot 3 robot environment file
    Mainly based on OpenAI code taken from https://bitbucket.org/theconstructcore/openai_ros
    Added few fixes, extended documentation

    some thoghts taken from https://github.com/karray/neuroracer/blob/master/neuroracer_gym/src/neuroracer_gym/neuroracer_env.py

    As of now, no camera feed is subscribed to 
    and passed to the training model for processing. 
    The goal was to have a very reasonably lightweight environment

    IMU topic is removed from this implementation for now
'''
import time
import numpy as np

import rospy
from openai_ros import robot_gazebo_env

# Modelstate service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


# sensor readings
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from helpers import helper_methods

DEFAULT_WAIT_TIME = 1


class T3LazyRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    '''
        Initializes a new TurtleBot3Env environment.
        TurtleBot3 doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion (at all for now). 

        To check any topic we need to have the simulations running, we need to do two things:
            1) Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
            that are paused for whatever the reason
            2) If the simulation was running already for some reason, we need to reset the controllers.
            This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
            and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot        
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel

    '''

    def __init__(self):
        '''
            Constructor for the robot environment
            Sets up default values and passes them to the parent gazebo env
        '''

        # used to set up T3's initial position when resetting the environment
        self.initial_position = None

        # the robot should avoid objects closer than 0.2m
        # this also means that to pass, the robot needs a >0.4m space
        # change this value to control space needed by the turtlebot
        self.minimum_distance = 0.2

        # list of controllers to pass to the gazebo env - Turtlebot3 does not use any
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(T3LazyRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                             robot_name_space=self.robot_name_space,
                                             reset_controls=False,
                                             start_init_physics_parameters=False)

        # checking for gazebo service
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

        self.gazebo.unpauseSim()
        time.sleep(DEFAULT_WAIT_TIME)

        # self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # subscribe to relevant topics and create cmd_vel publisher
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)

        # queue size set to 1 as only the latest message is relevant.
        # If for whatever reason a message is lost, a new one will quicky be sent with an updated value anyway
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished T3LazyRobotEnv INIT...")

    def reset_position(self):
        if not self.initial_position:
            return
        state_msg = ModelState()
        state_msg.model_name = 'racecar'
        state_msg.pose.position.x = self.initial_position['p_x']
        state_msg.pose.position.y = self.initial_position['p_y']
        state_msg.pose.position.z = self.initial_position['p_z']
        state_msg.pose.orientation.x = self.initial_position['o_x']
        state_msg.pose.orientation.y = self.initial_position['o_y']
        state_msg.pose.orientation.z = self.initial_position['o_z']
        state_msg.pose.orientation.w = self.initial_position['o_w']

        self.set_model_state(state_msg)

    def reset(self):
        '''
            Resets the environment, incluiding the robot's position
        '''
        super(T3LazyRobotEnv, self).reset()
        self.gazebo.unpauseSim()
        self.reset_position()

        time.sleep(DEFAULT_WAIT_TIME)
        self.gazebo.pauseSim()

        return self._get_obs()

    # system readiness checks
    def _check_all_sensors_ready(self):
        '''
            Checks if all required sensors are operational
            currently only checks for laser scan and odom                        
        '''
        rospy.logdebug("START ALL SENSORS READY")
        self._check_laser_scan_ready()
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(
                    "/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr(
                    "Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(
                    "/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr(
                    "Current /odom not ready yet, retrying for getting odom")

        return self.odom

    # callback methods
    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _odom_callback(self, data):
        self.odom = data

    def _check_publishers_connection(self):
        '''
            Checks that all the publishers are working
            :return:
        '''
        # 10hz -> if changing this make sure to change the queue_size when publishing to cmd_vel
        rate = rospy.Rate(10)

        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug(
                "No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # additional private methods

    def get_laser_scan(self):
        '''
            Returns the latest laser scan readings as a numpy array
        '''
        return np.array(self.laser_scan.ranges, dtype=np.float32)

    def _is_collided(self):
        '''
            Checks if the robot has collided with an obstacle or not
            taken from https://github.com/karray/neuroracer/blob/master/neuroracer_gym/src/neuroracer_gym/neuroracer_env.py

            collision is checked on getting a reading below the minimum distance
            but to handle potential errors, the mean of the 10 laser readings in the vicinity are checked
            if the mean is also below the minimum, the robot collided
        '''
        r = self.get_laser_scan()
        crashed = np.any(r <= self.min_distance)
        if crashed:
            min_range_idx = r.argmin()
            min_idx = min_range_idx - 5
            if min_idx < 0:
                min_idx = 0
            max_idx = min_idx + 10
            if max_idx >= r.shape[0]:
                max_idx = r.shape[0] - 1
                min_idx = max_idx - 10
            mean_distance = r[min_idx:max_idx].mean()

            crashed = np.any(mean_distance <= self.min_distance)

        return crashed

    # Methods that the Training Environment will need
    # some of them are defined here as 'virtual'
    # because they will be used in RobotGazeboEnv parent class and defined in the
    # TrainingEnvironment.
    # so those areare essentially a list of pass-through defintions for virtual methods
    # ----------------------------
    def _set_init_pose(self):
        '''
            Sets the Robot in its init pose
        '''
        raise NotImplementedError()

    def _init_env_variables(self):
        '''
            Inits variables needed to be initialised each time we reset at the start of an episode.
        '''
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        '''
            Calculates the reward to give based on the observations given.
        '''
        raise NotImplementedError()

    def _set_action(self, action):
        '''
            Applies the given action to the simulation.
        '''
        raise NotImplementedError()

    def _get_obs(self):
        '''
            returns the observations by the robot, which is only the LIDAR for now
        '''
        return self.laser_scan

    # to be used in the training environment, not virtual
    def _is_done(self, observations):
        '''
            Checks if episode done based on observations given.
            The episode is done if the robot has collided 
        '''
        self._episode_done = self._is_collided()
        return self._episode_done

    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        '''
            It will move the base based on the linear and angular speeds given.
            It will wait until those twists are achived reading from the odometry topic.
            :param linear_speed: Speed in the X axis of the robot base frame
            :param angular_speed: Speed of the angular turning of the robot base frame
            :param epsilon: Acceptable difference between the speed asked and the odometry readings
            :param update_rate: Rate at which we check the odometry.
            
            # v1.0 - no waiting for the odometry, we assume it is ok...
        '''
        cmd = CreateTwist(linear_speed, angular_speed)

        # TODO: this may be very unnecessary -> a try catch should suffice
        # self._check_publishers_connection()

        self._cmd_vel_pub.publish(cmd)

        # TODO: check if we need this
        # self.wait_until_twist_achieved(cmd_vel_value,
        #                               epsilon,
        #                               update_rate)

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        '''
            We wait for the cmd_vel twist given to be reached by the robot reading
            from the odometry.
            :param cmd_vel_value: Twist we want to wait to reach.
            :param epsilon: Error acceptable in odometry readings.
            :param update_rate: Rate at which we check the odometry.
            :return:
        '''
        rospy.logdebug("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            # IN turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = -1*current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) +
                           ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) +
                           ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (
                odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (
                odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time - start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logdebug("END wait_until_twist_achieved...")

        return delta_time
