#!/usr/bin/env python3
"""
Multi-Camera System Configuration
Defines hardware and software configurations for 11-camera warehouse system
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class SystemTier(Enum):
    """System performance tiers"""
    BUDGET = "budget"           # Single GPU, mixed models
    PERFORMANCE = "performance" # Single high-end GPU
    ENTERPRISE = "enterprise"   # Multi-GPU setup

class DetectionModel(Enum):
    """Available detection models"""
    GROUNDING_DINO = "grounding_dino"  # High accuracy, slower
    YOLOV8N = "yolov8n"               # Fast, good accuracy
    YOLOV8S = "yolov8s"               # Medium speed/accuracy
    YOLOV8M = "yolov8m"               # Higher accuracy, slower

@dataclass
class CameraConfig:
    """Configuration for individual camera"""
    camera_id: int
    priority: str  # "high", "medium", "low"
    model: DetectionModel
    resolution: Tuple[int, int]  # (width, height)
    detection_rate: float  # Hz
    gpu_id: int
    worker_threads: int
    batch_size: int
    use_fp16: bool

@dataclass
class HardwareConfig:
    """Hardware configuration specifications"""
    tier: SystemTier
    gpu_count: int
    gpu_memory_gb: List[int]
    cpu_cores: int
    ram_gb: int
    network_bandwidth_gbps: float
    storage_type: str

@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    max_gpu_memory_fraction: float
    use_mixed_precision: bool
    enable_tensorrt: bool
    queue_size: int
    frame_skip_ratio: int
    adaptive_batching: bool
    load_balancing: bool

class MultiCameraSystemConfig:
    """Multi-camera system configuration manager"""
    
    # Hardware Configurations
    HARDWARE_CONFIGS = {
        SystemTier.BUDGET: HardwareConfig(
            tier=SystemTier.BUDGET,
            gpu_count=1,
            gpu_memory_gb=[16],  # RTX 4080
            cpu_cores=16,
            ram_gb=32,
            network_bandwidth_gbps=1.0,
            storage_type="NVMe SSD 1TB"
        ),
        
        SystemTier.PERFORMANCE: HardwareConfig(
            tier=SystemTier.PERFORMANCE,
            gpu_count=1,
            gpu_memory_gb=[24],  # RTX 4090
            cpu_cores=24,
            ram_gb=64,
            network_bandwidth_gbps=10.0,
            storage_type="NVMe SSD 2TB"
        ),
        
        SystemTier.ENTERPRISE: HardwareConfig(
            tier=SystemTier.ENTERPRISE,
            gpu_count=3,
            gpu_memory_gb=[24, 24, 24],  # 3x RTX 4090 or 2x RTX 6000
            cpu_cores=32,
            ram_gb=128,
            network_bandwidth_gbps=10.0,
            storage_type="NVMe SSD 4TB RAID"
        )
    }
    
    # Camera Layout (from your warehouse setup)
    CAMERA_LAYOUT = {
        # Column 1 (Cameras 1-4)
        1: {"area": (10, 70, 0, 22.5), "priority": "high"},
        2: {"area": (10, 70, 22.5, 45), "priority": "high"}, 
        3: {"area": (10, 70, 45, 67.5), "priority": "high"},
        4: {"area": (10, 70, 67.5, 90), "priority": "high"},
        
        # Column 2 (Cameras 5-8) 
        5: {"area": (70, 130, 0, 22.5), "priority": "medium"},
        6: {"area": (70, 130, 22.5, 45), "priority": "medium"},
        7: {"area": (70, 130, 45, 67.5), "priority": "medium"},
        8: {"area": (70, 130, 67.5, 90), "priority": "medium"},
        
        # Column 3 (Cameras 9-11)
        9: {"area": (130, 180, 0, 30), "priority": "low"},
        10: {"area": (130, 180, 30, 60), "priority": "low"},
        11: {"area": (130, 180, 60, 90), "priority": "low"}
    }
    
    @classmethod
    def get_budget_config(cls) -> Dict[int, CameraConfig]:
        """Budget configuration: Single GPU, mixed models"""
        configs = {}
        
        for camera_id, layout in cls.CAMERA_LAYOUT.items():
            priority = layout["priority"]
            
            # High priority cameras get better models
            if priority == "high":
                model = DetectionModel.GROUNDING_DINO
                resolution = (1920, 1080)  # Reduced resolution
                detection_rate = 2.0
                worker_threads = 2
                batch_size = 1
            elif priority == "medium":
                model = DetectionModel.YOLOV8S
                resolution = (1920, 1080)
                detection_rate = 3.0
                worker_threads = 1
                batch_size = 2
            else:  # low priority
                model = DetectionModel.YOLOV8N
                resolution = (1280, 720)
                detection_rate = 4.0
                worker_threads = 1
                batch_size = 3
            
            configs[camera_id] = CameraConfig(
                camera_id=camera_id,
                priority=priority,
                model=model,
                resolution=resolution,
                detection_rate=detection_rate,
                gpu_id=0,  # All on single GPU
                worker_threads=worker_threads,
                batch_size=batch_size,
                use_fp16=True
            )
        
        return configs
    
    @classmethod
    def get_performance_config(cls) -> Dict[int, CameraConfig]:
        """Performance configuration: Single high-end GPU"""
        configs = {}
        
        for camera_id, layout in cls.CAMERA_LAYOUT.items():
            priority = layout["priority"]
            
            # All cameras can use good models
            if priority == "high":
                model = DetectionModel.GROUNDING_DINO
                resolution = (3840, 2160)  # Full 4K
                detection_rate = 3.0
                worker_threads = 3
                batch_size = 2
            elif priority == "medium":
                model = DetectionModel.GROUNDING_DINO
                resolution = (1920, 1080)  # 2K
                detection_rate = 4.0
                worker_threads = 2
                batch_size = 2
            else:  # low priority
                model = DetectionModel.YOLOV8S
                resolution = (1920, 1080)
                detection_rate = 5.0
                worker_threads = 2
                batch_size = 3
            
            configs[camera_id] = CameraConfig(
                camera_id=camera_id,
                priority=priority,
                model=model,
                resolution=resolution,
                detection_rate=detection_rate,
                gpu_id=0,  # Single GPU
                worker_threads=worker_threads,
                batch_size=batch_size,
                use_fp16=True
            )
        
        return configs
    
    @classmethod
    def get_enterprise_config(cls) -> Dict[int, CameraConfig]:
        """Enterprise configuration: Multi-GPU setup"""
        configs = {}
        
        # GPU distribution
        gpu_assignment = {
            1: 0, 2: 0, 3: 0, 4: 0,    # GPU 0: Cameras 1-4
            5: 1, 6: 1, 7: 1, 8: 1,    # GPU 1: Cameras 5-8  
            9: 2, 10: 2, 11: 2         # GPU 2: Cameras 9-11
        }
        
        for camera_id, layout in cls.CAMERA_LAYOUT.items():
            priority = layout["priority"]
            gpu_id = gpu_assignment[camera_id]
            
            # All cameras get maximum performance
            configs[camera_id] = CameraConfig(
                camera_id=camera_id,
                priority=priority,
                model=DetectionModel.GROUNDING_DINO,
                resolution=(3840, 2160),  # Full 4K for all
                detection_rate=5.0,       # High detection rate
                gpu_id=gpu_id,
                worker_threads=4,         # Maximum workers
                batch_size=2,
                use_fp16=True
            )
        
        return configs
    
    @classmethod
    def get_performance_settings(cls, tier: SystemTier) -> PerformanceConfig:
        """Get performance settings for tier"""
        if tier == SystemTier.BUDGET:
            return PerformanceConfig(
                max_gpu_memory_fraction=0.85,
                use_mixed_precision=True,
                enable_tensorrt=False,
                queue_size=3,
                frame_skip_ratio=2,  # Skip every 2nd frame
                adaptive_batching=True,
                load_balancing=False
            )
        elif tier == SystemTier.PERFORMANCE:
            return PerformanceConfig(
                max_gpu_memory_fraction=0.9,
                use_mixed_precision=True,
                enable_tensorrt=True,
                queue_size=5,
                frame_skip_ratio=0,  # No frame skipping
                adaptive_batching=True,
                load_balancing=True
            )
        else:  # ENTERPRISE
            return PerformanceConfig(
                max_gpu_memory_fraction=0.95,
                use_mixed_precision=True,
                enable_tensorrt=True,
                queue_size=8,
                frame_skip_ratio=0,
                adaptive_batching=True,
                load_balancing=True
            )
    
    @classmethod
    def estimate_performance(cls, tier: SystemTier) -> Dict[str, float]:
        """Estimate system performance for tier"""
        hardware = cls.HARDWARE_CONFIGS[tier]
        
        if tier == SystemTier.BUDGET:
            return {
                "total_fps": 25.0,
                "avg_detection_latency_ms": 400,
                "gpu_utilization_percent": 85,
                "estimated_accuracy": 0.82
            }
        elif tier == SystemTier.PERFORMANCE:
            return {
                "total_fps": 45.0,
                "avg_detection_latency_ms": 250,
                "gpu_utilization_percent": 90,
                "estimated_accuracy": 0.88
            }
        else:  # ENTERPRISE
            return {
                "total_fps": 75.0,
                "avg_detection_latency_ms": 150,
                "gpu_utilization_percent": 95,
                "estimated_accuracy": 0.92
            }
    
    @classmethod
    def print_configuration_summary(cls, tier: SystemTier):
        """Print detailed configuration summary"""
        print(f"\n🏗️ {tier.value.upper()} CONFIGURATION SUMMARY")
        print("=" * 60)
        
        # Hardware
        hardware = cls.HARDWARE_CONFIGS[tier]
        print(f"🖥️  Hardware:")
        print(f"   GPUs: {hardware.gpu_count}x {hardware.gpu_memory_gb}GB")
        print(f"   CPU: {hardware.cpu_cores} cores")
        print(f"   RAM: {hardware.ram_gb}GB")
        print(f"   Network: {hardware.network_bandwidth_gbps} Gbps")
        print(f"   Storage: {hardware.storage_type}")
        
        # Camera config
        if tier == SystemTier.BUDGET:
            camera_configs = cls.get_budget_config()
        elif tier == SystemTier.PERFORMANCE:
            camera_configs = cls.get_performance_config()
        else:
            camera_configs = cls.get_enterprise_config()
        
        print(f"\n📹 Camera Configuration:")
        for camera_id, config in camera_configs.items():
            res = f"{config.resolution[0]}x{config.resolution[1]}"
            print(f"   Camera {camera_id}: {config.model.value} | {res} | {config.detection_rate}Hz | GPU{config.gpu_id}")
        
        # Performance
        perf = cls.estimate_performance(tier)
        print(f"\n📊 Estimated Performance:")
        print(f"   Total FPS: {perf['total_fps']}")
        print(f"   Latency: {perf['avg_detection_latency_ms']}ms")
        print(f"   GPU Util: {perf['gpu_utilization_percent']}%")
        print(f"   Accuracy: {perf['estimated_accuracy']:.1%}")
        
        # Cost estimate
        if tier == SystemTier.BUDGET:
            cost = "$8,000 - $12,000"
        elif tier == SystemTier.PERFORMANCE:
            cost = "$15,000 - $20,000"
        else:
            cost = "$30,000 - $45,000"
        
        print(f"\n💰 Estimated Hardware Cost: {cost}")

def main():
    """Demo configuration options"""
    print("🏗️ MULTI-CAMERA SYSTEM CONFIGURATION OPTIONS")
    print("=" * 70)
    
    for tier in SystemTier:
        MultiCameraSystemConfig.print_configuration_summary(tier)
        print()

if __name__ == "__main__":
    main() 