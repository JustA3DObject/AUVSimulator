import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from rclpy.qos import qos_profile_sensor_data

class LAUVController(Node):
    def __init__(self):
        super().__init__('lauv_controller')
        
        # 1. Publishers
        self.thruster_pub = self.create_publisher(Float64MultiArray, '/lauv/thruster/setpoint', 10)
        self.fin_pub = self.create_publisher(JointState, '/lauv/fins/setpoint', 10)
        
        # 2. Subscriber (Using Sensor Data QoS to match Stonefish)
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/lauv/dynamics/odometry', 
            self.odom_callback, 
            qos_profile_sensor_data
        )
        
        # 3. State Variables
        self.odom_received = False
        
        # Tracking Position and Speed
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.vel_x = 0.0 # Forward/backward velocity relative to the AUV
        self.speed = 0.0 # Absolute scalar speed
        
        self.current_yaw = 0.0
        self.step = 0
        
        self.thrust_cmd = [0.0]
        self.fin_cmd = [0.0, 0.0, 0.0, 0.0]
        
        # 4. Timers
        # 10Hz loop to continuously publish commands and log data
        self.control_timer = self.create_timer(0.1, self.publish_commands)
        # 5-second loop to cycle through the test sequence
        self.sequence_timer = self.create_timer(5.0, self.update_sequence)
        
        self.get_logger().info('LAUV Controller Started. Waiting for simulator to unpause...')

    def odom_callback(self, msg):
        self.odom_received = True
        
        # Extract Position
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        self.pos_z = msg.pose.pose.position.z
        
        # Extract Orientation
        q = msg.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.current_yaw = yaw

        # Extract Velocity and calculate absolute speed
        self.vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel_z = msg.twist.twist.linear.z
        
        self.speed = math.sqrt(self.vel_x**2 + vel_y**2 + vel_z**2)

    def update_sequence(self):
        if not self.odom_received:
            return

        self.get_logger().info(f'\n--- SEQUENCE STEP {self.step} ---')

        if self.step == 0:
            self.get_logger().info('1. FORWARD (Base Thrust: 150.0)')
            self.thrust_cmd = [150.0]
            self.fin_cmd = [0.0, 0.0, 0.0, 0.0]

        elif self.step == 1:
            self.get_logger().info('2. FAST FORWARD (Increased Thrust: 250.0)')
            self.thrust_cmd = [250.0]
            self.fin_cmd = [0.0, 0.0, 0.0, 0.0]

        elif self.step == 2:
            self.get_logger().info('3. ACTIVE BRAKING (Thrust: -250.0 while moving forward)')
            # Throwing it in full reverse to kill forward momentum quickly
            self.thrust_cmd = [-250.0] 
            self.fin_cmd = [0.0, 0.0, 0.0, 0.0]

        elif self.step == 3:
            self.get_logger().info('4. BACKWARD (High Reverse Thrust: -200.0)')
            # Overcoming the weaker reverse thrust coefficient in the XML
            self.thrust_cmd = [-200.0] 
            self.fin_cmd = [0.0, 0.0, 0.0, 0.0]

        elif self.step == 4:
            self.get_logger().info('5. LEFT (Yaw Fins Deflected)')
            self.thrust_cmd = [150.0]
            self.fin_cmd = [0.4, -0.4, 0.0, 0.0] 

        elif self.step == 5:
            self.get_logger().info('6. RIGHT (Yaw Fins Deflected)')
            self.thrust_cmd = [150.0]
            self.fin_cmd = [-0.4, 0.4, 0.0, 0.0] 

        elif self.step == 6:
            self.get_logger().info('7. UPWARD (Pitch Fins Deflected)')
            self.thrust_cmd = [150.0]
            self.fin_cmd = [0.0, 0.0, 0.4, -0.4] 

        elif self.step == 7:
            self.get_logger().info('8. DOWNWARD (Pitch Fins Deflected)')
            self.thrust_cmd = [150.0]
            self.fin_cmd = [0.0, 0.0, -0.4, 0.4] 

        elif self.step == 8:
            self.get_logger().info('9. PASSIVE STOPPING (Thrust: 0.0)')
            self.thrust_cmd = [0.0]
            self.fin_cmd = [0.0, 0.0, 0.0, 0.0]
            self.step = -1

        self.step += 1

    def publish_commands(self):
        if not self.odom_received:
            return

        # Print position and speed at every control time step
        self.get_logger().info(
            f'Pos: [X: {self.pos_x:.2f}, Y: {self.pos_y:.2f}, Z: {self.pos_z:.2f}] | '
            f'Speed: {self.speed:.2f} m/s (Vx: {self.vel_x:.2f}) | Thrust Cmd: {self.thrust_cmd[0]}'
        )

        # Publish thrust
        thrust_msg = Float64MultiArray()
        thrust_msg.data = self.thrust_cmd
        self.thruster_pub.publish(thrust_msg)

        # Publish fins 
        fin_msg = JointState()
        fin_msg.name = ['lauv/FinTop', 'lauv/FinBottom', 'lauv/FinStbd', 'lauv/FinPort']
        fin_msg.position = self.fin_cmd
        self.fin_pub.publish(fin_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LAUVController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Test stopped by user.')
    finally:
        # Safe shutdown
        stop_thrust = Float64MultiArray(data=[0.0])
        stop_fins = JointState()
        stop_fins.name = ['lauv/FinTop', 'lauv/FinBottom', 'lauv/FinStbd', 'lauv/FinPort']
        stop_fins.position = [0.0, 0.0, 0.0, 0.0]
        
        node.thruster_pub.publish(stop_thrust)
        node.fin_pub.publish(stop_fins)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()