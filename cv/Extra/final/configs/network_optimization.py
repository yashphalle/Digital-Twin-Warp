#!/usr/bin/env python3
"""
Network Optimization Configuration for Multi-Camera RTSP
Addresses corruption and bandwidth issues with 11 cameras
"""

import os
import subprocess
import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

@dataclass
class NetworkOptimizationConfig:
    """Network optimization settings for multi-camera RTSP"""
    
    # Bandwidth management
    max_total_bandwidth_mbps: int = 80  # Total bandwidth limit
    per_camera_bandwidth_mbps: int = 6  # Per camera bandwidth limit
    
    # Connection settings
    tcp_buffer_size: int = 262144  # 256KB TCP buffer
    udp_buffer_size: int = 262144  # 256KB UDP buffer
    socket_timeout: int = 5  # Socket timeout in seconds
    
    # RTSP specific
    rtsp_transport: str = "tcp"  # tcp, udp, or auto
    rtsp_latency_ms: int = 100  # RTSP latency in milliseconds
    rtsp_buffer_size: int = 1  # RTSP buffer size
    
    # Quality settings
    resolution_limit: tuple = (1280, 720)  # Max resolution per camera
    fps_limit: int = 15  # Max FPS per camera
    quality_preset: str = "medium"  # low, medium, high
    
    # Retry and recovery
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 2.0
    health_check_interval: int = 30  # seconds
    
    # Threading
    max_concurrent_connections: int = 11
    connection_stagger_delay: float = 0.5  # seconds between connections


class NetworkOptimizer:
    """
    Network optimization utilities for multi-camera RTSP streaming
    """
    
    def __init__(self, config: NetworkOptimizationConfig = None):
        self.config = config or NetworkOptimizationConfig()
        logger.info("🌐 Network Optimizer initialized")
    
    def optimize_system_settings(self) -> bool:
        """
        Optimize system-level network settings for multi-camera streaming
        """
        logger.info("🔧 Optimizing system network settings...")
        
        optimizations_applied = []
        
        try:
            # Windows optimizations
            if os.name == 'nt':
                optimizations_applied.extend(self._optimize_windows_network())
            
            # Linux optimizations
            elif os.name == 'posix':
                optimizations_applied.extend(self._optimize_linux_network())
            
            if optimizations_applied:
                logger.info("✅ Network optimizations applied:")
                for opt in optimizations_applied:
                    logger.info(f"   - {opt}")
                return True
            else:
                logger.warning("⚠️ No network optimizations could be applied")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to optimize network settings: {e}")
            return False
    
    def _optimize_windows_network(self) -> List[str]:
        """Apply Windows-specific network optimizations"""
        optimizations = []
        
        try:
            # Increase TCP receive window
            cmd = "netsh int tcp set global autotuninglevel=normal"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                optimizations.append("TCP auto-tuning enabled")
            
            # Optimize TCP chimney
            cmd = "netsh int tcp set global chimney=enabled"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                optimizations.append("TCP chimney offload enabled")
            
            # Set RSS (Receive Side Scaling)
            cmd = "netsh int tcp set global rss=enabled"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                optimizations.append("RSS enabled")
                
        except Exception as e:
            logger.warning(f"⚠️ Windows network optimization warning: {e}")
        
        return optimizations
    
    def _optimize_linux_network(self) -> List[str]:
        """Apply Linux-specific network optimizations"""
        optimizations = []
        
        try:
            # Increase network buffer sizes
            sysctl_settings = [
                ("net.core.rmem_max", "134217728"),  # 128MB
                ("net.core.wmem_max", "134217728"),  # 128MB
                ("net.core.rmem_default", "262144"),  # 256KB
                ("net.core.wmem_default", "262144"),  # 256KB
                ("net.ipv4.tcp_rmem", "4096 262144 134217728"),
                ("net.ipv4.tcp_wmem", "4096 262144 134217728"),
                ("net.core.netdev_max_backlog", "5000"),
            ]
            
            for setting, value in sysctl_settings:
                try:
                    cmd = f"sudo sysctl -w {setting}={value}"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        optimizations.append(f"Set {setting}={value}")
                except:
                    pass  # Continue with other settings
                    
        except Exception as e:
            logger.warning(f"⚠️ Linux network optimization warning: {e}")
        
        return optimizations
    
    def get_optimal_rtsp_urls(self, base_urls: Dict[int, str]) -> Dict[int, str]:
        """
        Generate optimized RTSP URLs with transport and quality parameters
        """
        optimized_urls = {}
        
        for camera_id, base_url in base_urls.items():
            # Add transport parameter
            if "?" in base_url:
                optimized_url = f"{base_url}&tcp"
            else:
                optimized_url = f"{base_url}?tcp"
            
            # Add additional parameters for some camera types
            if "Streaming/channels/1" in base_url:
                # Lorex camera optimizations
                optimized_url += "&profile=main&level=3.1"
            
            optimized_urls[camera_id] = optimized_url
            logger.info(f"📹 Camera {camera_id}: Optimized URL generated")
        
        return optimized_urls
    
    def calculate_bandwidth_allocation(self, num_cameras: int) -> Dict[str, float]:
        """
        Calculate optimal bandwidth allocation for cameras
        """
        total_bandwidth = self.config.max_total_bandwidth_mbps
        per_camera_max = self.config.per_camera_bandwidth_mbps
        
        # Calculate actual per-camera allocation
        if num_cameras * per_camera_max > total_bandwidth:
            # Need to reduce per-camera bandwidth
            actual_per_camera = total_bandwidth / num_cameras
            logger.warning(f"⚠️ Reducing per-camera bandwidth to {actual_per_camera:.1f} Mbps")
        else:
            actual_per_camera = per_camera_max
        
        # Calculate quality settings based on bandwidth
        if actual_per_camera >= 6:
            quality = "high"
            resolution = (1920, 1080)
            fps = 20
        elif actual_per_camera >= 4:
            quality = "medium"
            resolution = (1280, 720)
            fps = 15
        else:
            quality = "low"
            resolution = (960, 540)
            fps = 10
        
        allocation = {
            'total_bandwidth_mbps': total_bandwidth,
            'per_camera_bandwidth_mbps': actual_per_camera,
            'recommended_quality': quality,
            'recommended_resolution': resolution,
            'recommended_fps': fps,
            'total_cameras': num_cameras
        }
        
        logger.info(f"📊 Bandwidth allocation for {num_cameras} cameras:")
        logger.info(f"   Per camera: {actual_per_camera:.1f} Mbps")
        logger.info(f"   Quality: {quality}")
        logger.info(f"   Resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"   FPS: {fps}")
        
        return allocation
    
    def generate_connection_strategy(self, camera_ids: List[int]) -> Dict[str, any]:
        """
        Generate optimal connection strategy for multiple cameras
        """
        num_cameras = len(camera_ids)
        
        # Staggered connection timing
        stagger_delay = max(0.5, num_cameras * 0.1)  # Increase delay with more cameras
        
        # Group cameras for batch connection
        batch_size = min(3, num_cameras)  # Connect in batches of 3
        batches = [camera_ids[i:i + batch_size] for i in range(0, num_cameras, batch_size)]
        
        # Priority order (connect most important cameras first)
        priority_cameras = [1, 2, 8, 9, 10, 11]  # Adjust based on your setup
        priority_order = []
        remaining_cameras = []
        
        for camera_id in camera_ids:
            if camera_id in priority_cameras:
                priority_order.append(camera_id)
            else:
                remaining_cameras.append(camera_id)
        
        # Combine priority and remaining
        connection_order = priority_order + remaining_cameras
        
        strategy = {
            'connection_order': connection_order,
            'stagger_delay': stagger_delay,
            'batch_size': batch_size,
            'batches': batches,
            'total_estimated_time': len(batches) * batch_size * stagger_delay,
            'max_concurrent': min(self.config.max_concurrent_connections, num_cameras)
        }
        
        logger.info(f"🎯 Connection strategy for {num_cameras} cameras:")
        logger.info(f"   Connection order: {connection_order}")
        logger.info(f"   Stagger delay: {stagger_delay:.1f}s")
        logger.info(f"   Estimated connection time: {strategy['total_estimated_time']:.1f}s")
        
        return strategy
    
    def monitor_network_health(self, camera_stats: Dict[int, dict]) -> Dict[str, any]:
        """
        Monitor network health based on camera statistics
        """
        total_cameras = len(camera_stats)
        connected_cameras = sum(1 for stats in camera_stats.values() if stats.get('connected', False))
        
        # Calculate success rates
        success_rates = []
        for stats in camera_stats.values():
            if 'success_rate' in stats:
                success_rates.append(stats['success_rate'])
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Determine health status
        if connected_cameras == total_cameras and avg_success_rate >= 95:
            health_status = "excellent"
        elif connected_cameras >= total_cameras * 0.8 and avg_success_rate >= 85:
            health_status = "good"
        elif connected_cameras >= total_cameras * 0.6 and avg_success_rate >= 70:
            health_status = "fair"
        else:
            health_status = "poor"
        
        health_report = {
            'status': health_status,
            'connected_cameras': connected_cameras,
            'total_cameras': total_cameras,
            'connection_rate': (connected_cameras / total_cameras) * 100,
            'avg_success_rate': avg_success_rate,
            'recommendations': self._get_health_recommendations(health_status, camera_stats)
        }
        
        logger.info(f"🏥 Network health: {health_status.upper()}")
        logger.info(f"   Connected: {connected_cameras}/{total_cameras} ({health_report['connection_rate']:.1f}%)")
        logger.info(f"   Avg success rate: {avg_success_rate:.1f}%")
        
        return health_report
    
    def _get_health_recommendations(self, health_status: str, camera_stats: Dict[int, dict]) -> List[str]:
        """Get recommendations based on network health"""
        recommendations = []
        
        if health_status == "poor":
            recommendations.extend([
                "Reduce number of active cameras",
                "Lower resolution/FPS settings",
                "Check network bandwidth usage",
                "Restart problematic cameras"
            ])
        elif health_status == "fair":
            recommendations.extend([
                "Monitor bandwidth usage",
                "Consider staggered camera activation",
                "Check for network congestion"
            ])
        elif health_status == "good":
            recommendations.append("System performing well, monitor for stability")
        else:  # excellent
            recommendations.append("Optimal performance achieved")
        
        # Camera-specific recommendations
        for camera_id, stats in camera_stats.items():
            if not stats.get('connected', False):
                recommendations.append(f"Reconnect Camera {camera_id}")
            elif stats.get('success_rate', 100) < 80:
                recommendations.append(f"Investigate Camera {camera_id} connection quality")
        
        return recommendations


def apply_network_optimizations(camera_configs: Dict[int, dict]) -> bool:
    """
    Apply comprehensive network optimizations for multi-camera setup
    """
    optimizer = NetworkOptimizer()
    
    # Apply system optimizations
    system_optimized = optimizer.optimize_system_settings()
    
    # Calculate bandwidth allocation
    num_cameras = len(camera_configs)
    bandwidth_allocation = optimizer.calculate_bandwidth_allocation(num_cameras)
    
    # Generate connection strategy
    camera_ids = list(camera_configs.keys())
    connection_strategy = optimizer.generate_connection_strategy(camera_ids)
    
    logger.info("🚀 Network optimization complete")
    return system_optimized


if __name__ == "__main__":
    # Test network optimization
    print("🌐 Network Optimization for Multi-Camera RTSP")
    
    # Example camera configuration
    camera_configs = {i: {'name': f'Camera {i}'} for i in range(1, 12)}
    
    success = apply_network_optimizations(camera_configs)
    print(f"✅ Optimization {'successful' if success else 'completed with warnings'}")
