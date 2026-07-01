# Global Humanoid Robot Challenge 2026 仿真资产仓库

## 概述

本仓库包含 Global Humanoid Robot Challenge 2026 的仿真资产文件，包括 USD 模型、URDF 文件等。

## 目录结构

```
.
├── README.md                # 本文档
├── .gitattributes           # Git LFS 配置文件
└── resources/              # 仿真资产文件
    ├── Box/                   # 标准料盒
    ├── Box_blank/             # 空白料盒
    ├── Box_blank_blue/        # 蓝色空白料盒
    ├── Box_blank_gray/        # 灰色空白料盒
    ├── Box_blue/             # 蓝色料盒
    ├── Box_gray/             # 灰色料盒
    ├── Collected_28motor/     # 电机部件
    ├── Collected_ConveyorBelt/# 传送带
    ├── Collected_foam/        # 泡沫垫
    ├── Collected_s2_v1_ecbg/  # WalkerS2 模型
    ├── Collected_table_v2/      # 桌子模型
    ├── Collected_Task1_PartA_ori_color/  # Task1 Part A
    ├── Collected_Task1_PartA_red/         # Task1 Part A (红色)
    ├── Collected_Task2_Part_A/            # Task2 Part A
    ├── Collected_Task4/                  # Task4 相关资源
    ├── *.usd                  # 齿轮、轴承等部件
    └── s2.urdf               # WalkerS2 机器人 URDF
```

## 主要资产

### 机器人
- **WalkerS2**: `resources/Collected_s2_v1_ecbg/s2_v1.usd`
- **URDF**: `resources/s2.urdf`

### 场景元素
- **桌子**: `resources/Collected_table_v2/table_v2.usd`
- **料盒**: `resources/Box/`, `resources/Box_blank/` 等
- **传送带**: `resources/Collected_ConveyorBelt/`

### 任务部件
- **Task1**: `Collected_Task1_PartA_ori_color/`, `Collected_Task1_PartA_red/`
- **Task2**: `Collected_Task2_Part_A/`
- **Task3**: `Task3_Part_A.usd`
- **Task4**: `Collected_Task4/`, `task4_box_foam.usd`

### 零件
- **齿轮**: `14-15-M1-5.usd`
- **轴承**: `6901-12246.usd`
- **电机**: `Collected_28motor/`
- **减速机**: `NN-CHC型-减速机总装配体.usd`
- **轴**: `减速输入轴.usd`, `输出法兰齿.usd`, `固定法兰齿.usd`
- **轴承套**: `滚针轴承K20x24x10.usd`

## 使用说明

本仓库作为 Git 子模块使用，在主项目中通过 `assets/` 路径访问。

资产文件位于 `resources/` 子目录中，配置文件中的 `root_path` 应该指向 `../assets/resources/` 或根据项目结构调整。

## Git LFS

本仓库使用 Git LFS 管理大文件，相关配置在 `.gitattributes` 中。

