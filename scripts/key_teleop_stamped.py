#!/usr/bin/env python3
import sys
import time
from dataclasses import dataclass

import pygame
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped  # Stamped for Jackal's controller

@dataclass
class Speeds:
    linear_x: float = 0.45   # m/s forward/back
    angular_z: float = 0.4  # rad/s left/right turn

class KeyTeleopStamped(Node):
    def __init__(self,
                 topic='/jackal_velocity_controller/cmd_vel',
                 hz=30.0,
                 speeds: Speeds = Speeds(),
                 frame_id='base_link'):
        super().__init__('key_teleop_stamped')
        self.pub = self.create_publisher(TwistStamped, topic, 10)
        self.dt = 1.0 / hz
        self.speeds = speeds
        self.frame_id = frame_id

        pygame.init()
        pygame.display.set_caption('Jackal Key Teleop (hold to move)')
        
        ### MODIFIED ### - Increased window size for better layout
        self.screen = pygame.display.set_mode((480, 280)) 
        self.font = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 20)

        self.last_send = time.time()
        self.timer = self.create_timer(self.dt, self._tick)

    def _compose_twist(self, keys):
        """
        Map keys to velocities.
        - Forward/back:   W / S or Up / Down
        - Turn left/right: A / D or Left / Right
        """
        v = 0.0
        w = 0.0

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            v += self.speeds.linear_x
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            v -= self.speeds.linear_x
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            w += self.speeds.angular_z
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            w -= self.speeds.angular_z

        return v, w

    def _publish(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self.pub.publish(msg)

    ### MODIFIED ### - The entire draw function is updated for visual feedback
    def _draw(self, v, w, keys):
        self.screen.fill((245, 245, 245))

        # --- Draw Top Status Text ---
        lines = [
            "Click window, then hold keys to move. Release to stop.",
            f"lin_x: {v:+.2f} m/s   ang_z: {w:+.2f} rad/s   topic: {self.pub.topic_name}",
            "Press ESC or close window to quit."
        ]
        y = 10
        for text in lines:
            surf = self.font_small.render(text, True, (20, 20, 20))
            self.screen.blit(surf, (10, y))
            y += 22

        # --- Draw Key Indicators ---
        KEY_RELEASED_COLOR = (200, 200, 200)
        KEY_PRESSED_COLOR = (120, 120, 120)
        KEY_TEXT_COLOR = (0, 0, 0)
        KEY_SIZE = 50
        KEY_SPACING = 10
        
        # Define keys, their labels, and their screen positions
        key_map = {
            'W / ↑': {'keys': (pygame.K_w, pygame.K_UP), 'pos': (KEY_SPACING + KEY_SIZE, 100)},
            'A / ←': {'keys': (pygame.K_a, pygame.K_LEFT), 'pos': (0, 100 + KEY_SIZE + KEY_SPACING)},
            'S / ↓': {'keys': (pygame.K_s, pygame.K_DOWN), 'pos': (KEY_SPACING + KEY_SIZE, 100 + KEY_SIZE + KEY_SPACING)},
            'D / →': {'keys': (pygame.K_d, pygame.K_RIGHT), 'pos': (2 * (KEY_SPACING + KEY_SIZE), 100 + KEY_SIZE + KEY_SPACING)},
        }
        
        # Center the key layout horizontally
        layout_width = 3 * KEY_SIZE + 2 * KEY_SPACING
        offset_x = (self.screen.get_width() - layout_width) / 2

        for label, data in key_map.items():
            is_pressed = keys[data['keys'][0]] or keys[data['keys'][1]]
            bg_color = KEY_PRESSED_COLOR if is_pressed else KEY_RELEASED_COLOR
            
            # Position the rectangle
            rect = pygame.Rect(data['pos'][0] + offset_x, data['pos'][1], KEY_SIZE, KEY_SIZE)
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
            
            # Render and center the text inside the rectangle
            text_surf = self.font.render(label.split(' ')[0], True, KEY_TEXT_COLOR) # Just show 'W', 'A', etc.
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)
            
        pygame.display.flip()


    def _tick(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                rclpy.shutdown()
                return

        keys = pygame.key.get_pressed()
        v, w = self._compose_twist(keys)
        self._publish(v, w)
        
        ### MODIFIED ### - Pass the `keys` state into the draw function
        self._draw(v, w, keys)
        self.last_send = time.time()

def main():
    rclpy.init()
    topic = sys.argv[1] if len(sys.argv) > 1 else '/jackal_velocity_controller/cmd_vel'
    hz = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    node = KeyTeleopStamped(topic=topic, hz=hz)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        pygame.quit()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()