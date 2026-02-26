Intelligent Traffic Flow Analysis System
Overview

This project implements a spatiotemporal traffic intelligence system for vehicle detection, tracking, and real-time traffic flow estimation.

The system integrates:

YOLOv8 for object detection

Multi-object tracking (ByteTrack via Ultralytics)

Direction-aware line-crossing detection

Class-wise vehicle analytics (car, bus, truck)

Sliding-window traffic flow estimation (vehicles/minute)

Structured CSV event logging for data-driven analysis

The system is designed as a practical prototype for intelligent transportation systems (ITS) and urban mobility analysis.

Key Features

Real-time vehicle detection and tracking

Direction-aware counting (avoids double counting)

Per-class vehicle statistics

Traffic flow estimation (vehicles/minute)

CSV-based structured dataset generation

Modular analytics architecture (encapsulation via analytics module)

Methodology

Detect vehicles using YOLOv8.

Track objects across frames with persistent IDs.

Detect line-crossing events using spatiotemporal position comparison.

Update class-wise counts.

Estimate flow rate using sliding-window analysis.

Log structured data for downstream analysis.

Example Output

Total vehicles counted

Cars, buses, trucks separately

Real-time flow rate (veh/min)

CSV file with timestamped event data

Research Motivation

This prototype demonstrates how deep learning-based perception systems can be integrated with temporal event modeling to generate structured traffic intelligence data for:

Urban traffic monitoring

Congestion analysis

Smart city infrastructure

Intelligent transportation systems research
