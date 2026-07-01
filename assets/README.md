# Global Humanoid Robot Challenge 2026 Simulation Asset Repository

## Overview

This repository contains simulation asset files for the Global Humanoid Robot Challenge 2026, including USD models, URDF files, and more.

## Directory Structure

```
.
├── README.md                
├── .gitattributes              # Git LFS configuration file
└── resources/                  # Simulation asset files
    ├── Box/                    # Standard material box
    ├── Box_blank/              # Blank material box
    ├── Box_blank_blue/         # Blue blank material box
    ├── Box_blank_gray/         # Gray blank material box
    ├── Box_blue/               # Blue material box
    ├── Box_gray/               # Gray material box
    ├── Collected_28motor/      # Motor components
    ├── Collected_ConveyorBelt/ # Conveyor belt
    ├── Collected_foam/         # Foam pad
    ├── Collected_s2_v1_ecbg/   # WalkerS2 model
    ├── Collected_table_v2/     # Table model
    ├── Collected_Task1_PartA_ori_color/    # Task1 Part A
    ├── Collected_Task1_PartA_red/          # Task1 Part A (Red)
    ├── Collected_Task2_Part_A/             # Task2 Part A
    ├── Collected_Task4/                    # Task4 related resources
    ├── *.usd                   # Gears, bearings, and other parts
    └── s2.urdf                 # WalkerS2 robot URDF
```

## Key Assets

### Robot
- **WalkerS2**: `resources/Collected_s2_v1_ecbg/s2_v1.usd`
- **URDF**: `resources/s2.urdf`

### Scene Elements
- **Table**: `resources/Collected_table_v2/table_v2.usd`
- **Material Boxes**: `resources/Box/`, `resources/Box_blank/`, etc.
- **Conveyor Belt**: `resources/Collected_ConveyorBelt/`

### Task Components
- **Task1**: `Collected_Task1_PartA_ori_color/`, `Collected_Task1_PartA_red/`
- **Task2**: `Collected_Task2_Part_A/`
- **Task3**: `Task3_Part_A.usd`
- **Task4**: `Collected_Task4/`, `task4_box_foam.usd`

### Parts
- **Gear**: `14-15-M1-5.usd`
- **Bearing**: `6901-12246.usd`
- **Motor**: `Collected_28motor/`
- **Gear Reducer**: `NN-CHC型-减速机总装配体.usd`
- **Shaft**: `减速输入轴.usd`, `输出法兰齿.usd`, `固定法兰齿.usd`
- **Bearing Sleeve**: `滚针轴承K20x24x10.usd`

## Usage Instructions

This repository is used as a Git submodule and is accessed via the `assets/` path in the main project.

Asset files are located in the `resources/` subdirectory. The `root_path` in the configuration file should point to `../assets/resources/` or be adjusted according to the project structure.

## Git LFS

This repository uses Git LFS to manage large files. The relevant configuration is in `.gitattributes`.
