# 项目配置

## 项目目标
地图经纬线检测 - 深度学习+传统方案混合

## Python 环境
- **解释器路径**: `/opt/homebrew/anaconda3/envs/qgis/bin/python`
- **Python 版本**: 3.12.12
- **运行命令**: `source activate.sh && python 脚本.py` 或直接 `/opt/homebrew/anaconda3/envs/qgis/bin/python 脚本.py`
- **关键包**: opencv 4.9.0, numpy 1.26.4

## 重要文件
以下文件是项目核心，在相关讨论时请参考：
- [demo.py](file:///Users/bytedance/Work/qgis_only/demo.py) - 传统算法入口
- [hybrid_pipeline.py](file:///Users/bytedance/Work/qgis_only/hybrid_pipeline.py) - 混合方案主程序
- [train_model.py](file:///Users/bytedance/Work/qgis_only/train_model.py) - 模型训练
- [data_preparation.py](file:///Users/bytedance/Work/qgis_only/data_preparation.py) - 数据准备
- [post_processing.py](file:///Users/bytedance/Work/qgis_only/post_processing.py) - 后处理模块
