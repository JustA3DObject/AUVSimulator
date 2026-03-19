#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class DockMover(Node):
    def __init__(self):
        super().__init__('dock_mover')
        
        # Create a publisher for the dock's thruster setpoint
        self.publisher_ = self.create_publisher(Float64, '/dock/thruster/setpoint', 500)
        
        # Publish at 10 Hz (0.1 seconds)
        timer_period = 0.1  
        self.timer = self.create_timer(timer_period, self.publish_thrust)
        
        self.get_logger().info("Starting dock movement in ROS 2...")

    def publish_thrust(self):
        msg = Float64()
        # Set normalized thrust value (-1.0 to 1.0)
        msg.data = 0.5  
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    dock_mover = DockMover()
    
    try:
        # Keep the node running and publishing
        rclpy.spin(dock_mover)
    except KeyboardInterrupt:
        dock_mover.get_logger().info("Dock mover node stopped cleanly.")
    finally:
        # Clean up
        dock_mover.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()