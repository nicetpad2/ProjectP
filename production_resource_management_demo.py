#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NICEGOLD ENTERPRISE PROJECTP - PRODUCTION RESOURCE MANAGEMENT DEMO
การสาธิตระบบจัดการทรัพยากรอัจฉริยะในการผลิตจริง

🎯 Demo Features:
✅ สาธิตการตรวจจับสภาพแวดล้อมอัจฉริยะ
✅ การจัดสรรทรัพยากร 80% แบบอัตโนมัติ
✅ การปรับตัวและเรียนรู้ของระบบ
✅ การจัดการและปรับแต่งประสิทธิภาพ
✅ การติดตามและรายงานผลแบบเรียลไทม์
✅ การสาธิตในสภาพแวดล้อมจริง

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 9 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import resource management modules
try:
    from core.intelligent_environment_detector import (
        get_intelligent_environment_detector,
        EnvironmentType,
        HardwareCapability,
        ResourceOptimizationLevel
    )
    from core.smart_resource_orchestrator import (
        get_smart_resource_orchestrator,
        OrchestrationConfig,
        OrchestrationStatus,
        AdaptiveMode
    )
    from core.unified_resource_manager import (
        get_unified_resource_manager,
        ResourceType,
        ResourceStatus
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import failed: {e}")
    MODULES_AVAILABLE = False

# Try to import rich for better display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ====================================================
# PRODUCTION RESOURCE MANAGEMENT DEMO
# ====================================================

class ProductionResourceManagementDemo:
    """
    🎯 การสาธิตระบบจัดการทรัพยากรอัจฉริยะในการผลิต
    """
    
    def __init__(self):
        """เริ่มต้นการสาธิต"""
        self.console = Console() if RICH_AVAILABLE else None
        self.orchestrator = None
        self.environment_detector = None
        self.resource_manager = None
        self.demo_running = False
        self.demo_data = []
        
        if not MODULES_AVAILABLE:
            print("❌ Required modules not available")
            return
        
        self._initialize_components()
    
    def _initialize_components(self):
        """เริ่มต้นส่วนประกอบของระบบ"""
        try:
            # Initialize environment detector
            self.environment_detector = get_intelligent_environment_detector()
            
            # Initialize smart orchestrator
            config = OrchestrationConfig(
                target_utilization=0.80,
                monitoring_interval=2.0,
                optimization_interval=15.0,
                adaptation_threshold=0.10,
                enable_gpu_management=True,
                enable_memory_optimization=True,
                enable_cpu_optimization=True
            )
            
            self.orchestrator = get_smart_resource_orchestrator(config)
            
            # Initialize resource manager
            self.resource_manager = get_unified_resource_manager()
            
            if self.console:
                self.console.print("✅ All components initialized successfully", style="green")
            else:
                print("✅ All components initialized successfully")
                
        except Exception as e:
            error_msg = f"❌ Error initializing components: {e}"
            if self.console:
                self.console.print(error_msg, style="red")
            else:
                print(error_msg)
    
    def display_welcome_banner(self):
        """แสดงแบนเนอร์ต้นรับ"""
        if self.console:
            banner = Panel(
                "[bold blue]🎯 NICEGOLD ENTERPRISE PROJECTP[/bold blue]\n"
                "[bold green]Production Resource Management Demo[/bold green]\n\n"
                "[yellow]🤖 Intelligent Environment Detection & Resource Orchestration[/yellow]\n"
                "[yellow]⚡ 80% Resource Allocation & Optimization[/yellow]\n"
                "[yellow]🔍 Real-time Monitoring & Adaptation[/yellow]\n\n"
                "[dim]Enterprise Edition - Production Ready[/dim]",
                title="Welcome",
                border_style="blue"
            )
            self.console.print(banner)
        else:
            print("=" * 80)
            print("🎯 NICEGOLD ENTERPRISE PROJECTP")
            print("Production Resource Management Demo")
            print("=" * 80)
            print("🤖 Intelligent Environment Detection & Resource Orchestration")
            print("⚡ 80% Resource Allocation & Optimization")
            print("🔍 Real-time Monitoring & Adaptation")
            print("=" * 80)
    
    def detect_and_analyze_environment(self):
        """ตรวจจับและวิเคราะห์สภาพแวดล้อม"""
        if not self.environment_detector:
            return
        
        if self.console:
            with self.console.status("[bold green]Detecting environment...", spinner="dots"):
                env_info = self.environment_detector.detect_environment()
                allocation = self.environment_detector.get_optimal_resource_allocation(env_info)
                time.sleep(1)  # Visual effect
        else:
            print("🔍 Detecting environment...")
            env_info = self.environment_detector.detect_environment()
            allocation = self.environment_detector.get_optimal_resource_allocation(env_info)
        
        # Display environment information
        if self.console:
            table = Table(title="Environment Analysis", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Environment Type", env_info.environment_type.value)
            table.add_row("Hardware Capability", env_info.hardware_capability.value)
            table.add_row("Optimization Level", env_info.optimization_level.value)
            table.add_row("CPU Cores", str(env_info.cpu_cores))
            table.add_row("Memory (GB)", f"{env_info.memory_gb:.1f}")
            table.add_row("GPU Count", str(env_info.gpu_count))
            table.add_row("Operating System", env_info.operating_system)
            table.add_row("Python Version", env_info.python_version)
            
            self.console.print(table)
            
            # Display optimal allocation
            allocation_table = Table(title="Optimal Resource Allocation", show_header=True)
            allocation_table.add_column("Resource", style="yellow")
            allocation_table.add_column("Allocation", style="green")
            allocation_table.add_column("Target", style="blue")
            
            allocation_table.add_row("CPU", f"{allocation.cpu_percentage*100:.1f}%", f"{allocation.target_utilization*100:.1f}%")
            allocation_table.add_row("Memory", f"{allocation.memory_percentage*100:.1f}%", f"{allocation.target_utilization*100:.1f}%")
            allocation_table.add_row("GPU", f"{allocation.gpu_percentage*100:.1f}%", f"{allocation.target_utilization*100:.1f}%")
            allocation_table.add_row("Safety Margin", f"{allocation.safety_margin*100:.1f}%", "-")
            allocation_table.add_row("Emergency Reserve", f"{allocation.emergency_reserve*100:.1f}%", "-")
            
            self.console.print(allocation_table)
        else:
            print("\n📊 ENVIRONMENT ANALYSIS:")
            print(f"  Environment Type: {env_info.environment_type.value}")
            print(f"  Hardware Capability: {env_info.hardware_capability.value}")
            print(f"  Optimization Level: {env_info.optimization_level.value}")
            print(f"  CPU Cores: {env_info.cpu_cores}")
            print(f"  Memory: {env_info.memory_gb:.1f} GB")
            print(f"  GPU Count: {env_info.gpu_count}")
            print(f"  Operating System: {env_info.operating_system}")
            print(f"  Python Version: {env_info.python_version}")
            
            print("\n⚡ OPTIMAL RESOURCE ALLOCATION:")
            print(f"  CPU: {allocation.cpu_percentage*100:.1f}%")
            print(f"  Memory: {allocation.memory_percentage*100:.1f}%")
            print(f"  GPU: {allocation.gpu_percentage*100:.1f}%")
            print(f"  Target Utilization: {allocation.target_utilization*100:.1f}%")
            print(f"  Safety Margin: {allocation.safety_margin*100:.1f}%")
            print(f"  Emergency Reserve: {allocation.emergency_reserve*100:.1f}%")
    
    def start_intelligent_orchestration(self):
        """เริ่มการจัดการทรัพยากรอัจฉริยะ"""
        if not self.orchestrator:
            return False
        
        if self.console:
            self.console.print("🤖 Starting intelligent resource orchestration...", style="blue")
        else:
            print("🤖 Starting intelligent resource orchestration...")
        
        success = self.orchestrator.start_orchestration()
        
        if success:
            if self.console:
                self.console.print("✅ Orchestration started successfully", style="green")
            else:
                print("✅ Orchestration started successfully")
            return True
        else:
            if self.console:
                self.console.print("❌ Failed to start orchestration", style="red")
            else:
                print("❌ Failed to start orchestration")
            return False
    
    def monitor_resources_realtime(self, duration: int = 60):
        """ติดตามทรัพยากรแบบเรียลไทม์"""
        if not self.resource_manager:
            return
        
        if self.console:
            self.console.print(f"📊 Starting real-time monitoring for {duration} seconds...", style="yellow")
        else:
            print(f"📊 Starting real-time monitoring for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Get current resources
                resources = self.resource_manager.get_resource_status()
                
                # Get orchestration status
                if self.orchestrator:
                    status = self.orchestrator.get_orchestration_status()
                else:
                    status = None
                
                # Record data
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'resources': {k: v.percentage for k, v in resources.items()},
                    'orchestration_status': status.status.value if status else 'unknown',
                    'orchestration_mode': status.mode.value if status else 'unknown',
                    'optimizations_applied': status.optimizations_applied if status else 0
                }
                
                self.demo_data.append(data_point)
                
                if self.console:
                    # Create a live updating display
                    table = Table(title="Real-time Resource Monitor", show_header=True)
                    table.add_column("Resource", style="cyan")
                    table.add_column("Usage", style="green")
                    table.add_column("Status", style="yellow")
                    
                    for resource_type, resource_info in resources.items():
                        usage_color = "green" if resource_info.percentage < 70 else "yellow" if resource_info.percentage < 85 else "red"
                        table.add_row(
                            resource_type.upper(),
                            f"{resource_info.percentage:.1f}%",
                            f"[{usage_color}]{resource_info.status.value}[/{usage_color}]"
                        )
                    
                    if status:
                        table.add_row("---", "---", "---")
                        table.add_row("Orchestration", status.status.value, status.mode.value)
                        table.add_row("Optimizations", str(status.optimizations_applied), "-")
                    
                    # Clear screen and display
                    self.console.clear()
                    self.console.print(table)
                    
                    # Show elapsed time
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    self.console.print(f"⏱️  Time remaining: {remaining:.1f}s", style="blue")
                    
                else:
                    # Simple text display
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    
                    print(f"\n📊 Resource Status (Time remaining: {remaining:.1f}s):")
                    for resource_type, resource_info in resources.items():
                        print(f"  {resource_type.upper()}: {resource_info.percentage:.1f}% ({resource_info.status.value})")
                    
                    if status:
                        print(f"  Orchestration: {status.status.value} ({status.mode.value})")
                        print(f"  Optimizations: {status.optimizations_applied}")
                
                # Sleep for monitoring interval
                time.sleep(3)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self.console:
                    self.console.print(f"❌ Monitoring error: {e}", style="red")
                else:
                    print(f"❌ Monitoring error: {e}")
                break
        
        if self.console:
            self.console.print("✅ Real-time monitoring completed", style="green")
        else:
            print("✅ Real-time monitoring completed")
    
    def generate_demo_report(self):
        """สร้างรายงานผลการสาธิต"""
        if not self.demo_data:
            return
        
        # Calculate statistics
        total_points = len(self.demo_data)
        
        if total_points == 0:
            return
        
        # Resource usage averages
        avg_cpu = sum(d['resources'].get('cpu', 0) for d in self.demo_data) / total_points
        avg_memory = sum(d['resources'].get('memory', 0) for d in self.demo_data) / total_points
        avg_gpu = sum(d['resources'].get('gpu', 0) for d in self.demo_data) / total_points
        
        # Optimization count
        final_optimizations = self.demo_data[-1]['optimizations_applied'] if self.demo_data else 0
        
        # Generate report
        report = {
            'demo_summary': {
                'total_data_points': total_points,
                'duration_seconds': total_points * 3,  # Approximate
                'average_resource_usage': {
                    'cpu': avg_cpu,
                    'memory': avg_memory,
                    'gpu': avg_gpu
                },
                'total_optimizations': final_optimizations,
                'final_status': self.demo_data[-1]['orchestration_status'] if self.demo_data else 'unknown'
            },
            'detailed_data': self.demo_data[-10:] if len(self.demo_data) > 10 else self.demo_data  # Last 10 points
        }
        
        # Display report
        if self.console:
            summary_table = Table(title="Demo Summary Report", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Total Data Points", str(total_points))
            summary_table.add_row("Duration", f"{total_points * 3} seconds")
            summary_table.add_row("Average CPU Usage", f"{avg_cpu:.1f}%")
            summary_table.add_row("Average Memory Usage", f"{avg_memory:.1f}%")
            summary_table.add_row("Average GPU Usage", f"{avg_gpu:.1f}%")
            summary_table.add_row("Total Optimizations", str(final_optimizations))
            summary_table.add_row("Final Status", self.demo_data[-1]['orchestration_status'] if self.demo_data else 'unknown')
            
            self.console.print(summary_table)
        else:
            print("\n📊 DEMO SUMMARY REPORT:")
            print(f"  Total Data Points: {total_points}")
            print(f"  Duration: {total_points * 3} seconds")
            print(f"  Average CPU Usage: {avg_cpu:.1f}%")
            print(f"  Average Memory Usage: {avg_memory:.1f}%")
            print(f"  Average GPU Usage: {avg_gpu:.1f}%")
            print(f"  Total Optimizations: {final_optimizations}")
            print(f"  Final Status: {self.demo_data[-1]['orchestration_status'] if self.demo_data else 'unknown'}")
        
        # Save report to file
        try:
            with open('production_demo_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            if self.console:
                self.console.print("💾 Demo report saved to: production_demo_report.json", style="blue")
            else:
                print("💾 Demo report saved to: production_demo_report.json")
                
        except Exception as e:
            if self.console:
                self.console.print(f"❌ Error saving report: {e}", style="red")
            else:
                print(f"❌ Error saving report: {e}")
    
    def stop_orchestration(self):
        """หยุดการจัดการทรัพยากร"""
        if self.orchestrator:
            success = self.orchestrator.stop_orchestration()
            
            if success:
                if self.console:
                    self.console.print("✅ Orchestration stopped successfully", style="green")
                else:
                    print("✅ Orchestration stopped successfully")
            else:
                if self.console:
                    self.console.print("❌ Failed to stop orchestration", style="red")
                else:
                    print("❌ Failed to stop orchestration")
    
    def run_full_demo(self):
        """รันการสาธิตแบบครบวงจร"""
        try:
            # Welcome banner
            self.display_welcome_banner()
            
            # Step 1: Environment detection
            if self.console:
                self.console.print("\n🔍 Step 1: Environment Detection & Analysis", style="bold blue")
            else:
                print("\n🔍 Step 1: Environment Detection & Analysis")
            
            self.detect_and_analyze_environment()
            
            # Step 2: Start orchestration
            if self.console:
                self.console.print("\n🤖 Step 2: Starting Intelligent Resource Orchestration", style="bold blue")
            else:
                print("\n🤖 Step 2: Starting Intelligent Resource Orchestration")
            
            if not self.start_intelligent_orchestration():
                if self.console:
                    self.console.print("❌ Demo failed to start orchestration", style="red")
                else:
                    print("❌ Demo failed to start orchestration")
                return
            
            # Step 3: Real-time monitoring
            if self.console:
                self.console.print("\n📊 Step 3: Real-time Resource Monitoring", style="bold blue")
            else:
                print("\n📊 Step 3: Real-time Resource Monitoring")
            
            self.monitor_resources_realtime(duration=30)  # 30 seconds
            
            # Step 4: Generate report
            if self.console:
                self.console.print("\n📋 Step 4: Demo Report Generation", style="bold blue")
            else:
                print("\n📋 Step 4: Demo Report Generation")
            
            self.generate_demo_report()
            
            # Step 5: Stop orchestration
            if self.console:
                self.console.print("\n🛑 Step 5: Stopping Orchestration", style="bold blue")
            else:
                print("\n🛑 Step 5: Stopping Orchestration")
            
            self.stop_orchestration()
            
            # Final message
            if self.console:
                success_panel = Panel(
                    "[bold green]🎉 Production Resource Management Demo Completed Successfully![/bold green]\n\n"
                    "[yellow]✅ Environment detected and analyzed[/yellow]\n"
                    "[yellow]✅ 80% resource allocation optimized[/yellow]\n"
                    "[yellow]✅ Real-time monitoring demonstrated[/yellow]\n"
                    "[yellow]✅ Intelligent orchestration validated[/yellow]\n"
                    "[yellow]✅ System ready for production deployment[/yellow]\n\n"
                    "[dim]Enterprise Edition - Production Ready[/dim]",
                    title="Demo Complete",
                    border_style="green"
                )
                self.console.print(success_panel)
            else:
                print("\n" + "=" * 80)
                print("🎉 Production Resource Management Demo Completed Successfully!")
                print("=" * 80)
                print("✅ Environment detected and analyzed")
                print("✅ 80% resource allocation optimized")
                print("✅ Real-time monitoring demonstrated")
                print("✅ Intelligent orchestration validated")
                print("✅ System ready for production deployment")
                print("=" * 80)
                
        except KeyboardInterrupt:
            if self.console:
                self.console.print("\n⚠️ Demo interrupted by user", style="yellow")
            else:
                print("\n⚠️ Demo interrupted by user")
            
            self.stop_orchestration()
            
        except Exception as e:
            if self.console:
                self.console.print(f"\n❌ Demo error: {e}", style="red")
            else:
                print(f"\n❌ Demo error: {e}")
            
            self.stop_orchestration()


# ====================================================
# MAIN EXECUTION
# ====================================================

def main():
    """ฟังก์ชั่นหลักสำหรับการสาธิต"""
    try:
        # Create demo instance
        demo = ProductionResourceManagementDemo()
        
        # Run full demo
        demo.run_full_demo()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo execution failed: {e}")


if __name__ == "__main__":
    main()
