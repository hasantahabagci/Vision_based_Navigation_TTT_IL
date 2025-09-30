#!/usr/bin/env python3
# Gerekli kütüphaneleri içe aktarıyoruz
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from geometry_msgs.msg import Pose, Point, Quaternion
import time
import math
import numpy as np

class CapsulePatrol(Node):
    """
    Gazebo'daki capsule modelini koridor boyunca sürekli hareket ettiren düğüm.
    """
    
    def __init__(self):
        # Düğümü başlatıyoruz
        super().__init__('capsule_patrol_node')
        
        # Servis istemcisini oluşturuyoruz
        self.client = self.create_client(
            SetEntityPose, 
            '/world/smooth_curved_corridor_with_rotations/set_pose'
        )
        
        # Servisin hazır olmasını bekliyoruz
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Servis bulunamadı, bekleniyor...')
        
        self.get_logger().info('Servis bulundu ve hazır!')
        
        # Waypoint'leri tanımlıyoruz (koridor boyunca noktalar)
        control_points = [
            (10.0, 3.5, 0.0),    # Başlangıç noktası (mevcut pozisyona yakın)
            (9.0, 3.5, 0.0),     # Sağ taraf
            (8.0, 3.5, 0.0),
            (7.0, 3.5, 0.0),
            (6.0, 3.5, 0.0),
            (5.0, 3.5, 0.0),
            (4.0, 3.0, 0.0),
            (3.0, 2.5, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.5, 0.0),
            (-1.0, -0.5, 0.0),
            (-2.0, -1.0, 0.0),
            (-3.0, -1.5, 0.0),
            (-4.0, -2.0, 0.0),
            (-5.0, -2.5, 0.0),
            (-6.0, -3.0, 0.0),
            (-7.0, -3.5, 0.0),
            (-8.0, -3.5, 0.0),
            (-9.0, -3.5, 0.0),   # Sol uç nokta
        ]
        self.waypoints = self.generate_spline_path(control_points, points_per_segment=40)
        # Mevcut waypoint indeksi
        self.current_waypoint_index = 0
        
        # Hareket yönü (1: ileri, -1: geri)
        self.direction = 1
        
        # Hareket hızı parametreleri
        self.move_interval = 0.01  # Her hareket arasındaki bekleme süresi (saniye)
        self.interpolation_steps = 10  # Her iki waypoint arası kaç adım
        
        # Timer oluştur
        self.timer = self.create_timer(self.move_interval, self.move_callback)
        
        # İnterpolasyon için değişkenler
        self.current_step = 0
        self.is_interpolating = False
        self.start_pos = None
        self.end_pos = None
        
        self.get_logger().info('Capsule patrol başladı!')

    def generate_spline_path(self, control_points, points_per_segment=20):
        """
        Catmull-Rom spline kullanarak kontrol noktalarından yumuşak bir yol oluşturur.
        """
        # Kontrol noktalarını numpy dizilerine çevir
        control_points = np.array(control_points)
        
        # Uç durumları yönetmek için başlangıç ve bitiş noktalarını tekrarla
        padded_points = np.vstack([
            control_points[0],
            control_points,
            control_points[-1]
        ])
        
        spline_points = []
        
        # Her segment için spline noktalarını hesapla
        for i in range(len(padded_points) - 3):
            p0 = padded_points[i]
            p1 = padded_points[i+1]
            p2 = padded_points[i+2]
            p3 = padded_points[i+3]
            
            for t in np.linspace(0, 1, points_per_segment, endpoint=False):
                # Catmull-Rom spline formülü
                point = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
                )
                spline_points.append(tuple(point))
                
        # Son noktayı eklediğimizden emin ol
        spline_points.append(tuple(control_points[-1]))
        
        return spline_points
    
    def calculate_rotation(self, current_pos, target_pos):
        """
        İki nokta arasındaki yönelimi hesapla (quaternion olarak)
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Yaw açısını hesapla (z ekseni etrafında dönüş)
        yaw = math.atan2(dy, dx)
        
        # Yaw açısını quaternion'a çevir
        w = math.cos(yaw / 2.0)
        x = 0.0
        y = 0.0
        z = math.sin(yaw / 2.0)
        
        return (x, y, z, w)
    
    def move_entity(self, name, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """
        Belirtilen modeli istenen koordinatlara taşı
        """
        request = SetEntityPose.Request()
        
        # Model bilgilerini ayarla
        request.entity = Entity()
        request.entity.name = name
        request.entity.type = Entity.MODEL
        
        # Pozisyon ve yönelimi ayarla
        request.pose = Pose()
        request.pose.position.x = x
        request.pose.position.y = y
        request.pose.position.z = z
        request.pose.orientation.x = qx
        request.pose.orientation.y = qy
        request.pose.orientation.z = qz
        request.pose.orientation.w = qw
        
        # Servisi asenkron olarak çağır
        future = self.client.call_async(request)
        return future
    
    def interpolate_position(self, start, end, t):
        """
        İki pozisyon arasında lineer interpolasyon yap
        t: 0.0 ile 1.0 arasında bir değer
        """
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        z = start[2] + (end[2] - start[2]) * t
        return (x, y, z)
    
    def move_callback(self):
        """
        Timer callback - her çağrıldığında capsule'ı hareket ettir
        """
        if not self.is_interpolating:
            # Yeni interpolasyon başlat
            if self.direction == 1:
                # İleri gidiyoruz
                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.start_pos = self.waypoints[self.current_waypoint_index]
                    self.end_pos = self.waypoints[self.current_waypoint_index + 1]
                    self.is_interpolating = True
                    self.current_step = 0
                else:
                    # Sona ulaştık, geri dön
                    self.direction = -1
                    self.get_logger().info('Koridorun sonuna ulaşıldı, geri dönülüyor...')
            else:
                # Geri gidiyoruz
                if self.current_waypoint_index > 0:
                    self.start_pos = self.waypoints[self.current_waypoint_index]
                    self.end_pos = self.waypoints[self.current_waypoint_index - 1]
                    self.is_interpolating = True
                    self.current_step = 0
                else:
                    # Başa döndük, tekrar ileri git
                    self.direction = 1
                    self.get_logger().info('Başlangıç noktasına ulaşıldı, tekrar ileri gidiliyor...')
        
        if self.is_interpolating:
            # İnterpolasyon yap
            t = self.current_step / float(self.interpolation_steps)
            current_pos = self.interpolate_position(self.start_pos, self.end_pos, t)
            
            # Yönelimi hesapla (hareket yönüne bak)
            quaternion = self.calculate_rotation(self.start_pos, self.end_pos)
            
            # Capsule'ı hareket ettir
            future = self.move_entity(
                'capsule', 
                current_pos[0], current_pos[1], current_pos[2],
                quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            )
            
            self.get_logger().debug(
                f'Capsule pozisyonu: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})'
            )
            
            # Adımı artır
            self.current_step += 1
            
            # İnterpolasyon tamamlandı mı?
            if self.current_step > self.interpolation_steps:
                self.is_interpolating = False
                # Waypoint indeksini güncelle
                self.current_waypoint_index += self.direction
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint_index}/{len(self.waypoints)-1} ulaşıldı'
                )

def main(args=None):
    # ROS 2'yi başlat
    rclpy.init(args=args)
    
    # Patrol düğümünü oluştur
    capsule_patrol = CapsulePatrol()
    
    try:
        # Düğümü çalıştır
        rclpy.spin(capsule_patrol)
    except KeyboardInterrupt:
        capsule_patrol.get_logger().info('Patrol durduruldu.')
    finally:
        # Temizlik
        capsule_patrol.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()