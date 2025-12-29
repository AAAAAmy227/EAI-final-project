---
description: 重构 SO101 Debug 打印为 logging 模块
---

# 任务：SO101 Debug 打印改为 logging

## 📋 任务概述

将 `scripts/so101.py` 中的 `print("DEBUG: ...")` 改为标准 `logging` 模块调用，提高代码专业性并支持日志级别控制。

## 🎯 目标

1. 在 `so101.py` 顶部添加 logger 初始化
2. 将 debug print 语句改为 `logger.debug()`
3. (可选) 将其他 print 语句改为适当的 logging 级别

---

## 📍 需要修改的文件

### 文件: `scripts/so101.py`

#### 修改点 1: 添加 logging 初始化 (文件顶部，约第 1-10 行)

**当前代码:**
```python
import numpy as np
import copy
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
# ...
```

**修改后:**
```python
import logging
import numpy as np
import copy
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
# ...

logger = logging.getLogger(__name__)
```

---

#### 修改点 2: `_sensor_configs` 属性 (约第 106-120 行)

**当前代码:**
```python
@property
def _sensor_configs(self):
    print("DEBUG: SO101._sensor_configs called")
    return [
        CameraConfig(
            # ...
        )
    ]
```

**修改后:**
```python
@property
def _sensor_configs(self):
    logger.debug("SO101._sensor_configs called")
    return [
        CameraConfig(
            # ...
        )
    ]
```

---

#### 修改点 3: `_after_loading_articulation` 中的 warning (约第 212-215 行)

**当前代码:**
```python
except KeyError:
    print("Warning: Fingertip links not found. TCP calculation will fall back to gripper links.")
    self.finger1_tip = self.finger1_link
    self.finger2_tip = self.finger2_link
```

**修改后:**
```python
except KeyError:
    logger.warning("Fingertip links not found. TCP calculation will fall back to gripper links.")
    self.finger1_tip = self.finger1_link
    self.finger2_tip = self.finger2_link
```

---

## ⚠️ 注意事项

1. **logger 命名**: 使用 `__name__` 获取模块名，便于日志过滤
2. **日志级别**: 
   - `debug`: 调试信息 (默认不显示)
   - `warning`: 警告信息 (需要注意但不是错误)
3. **不要修改**: 功能性的 print 语句 (如果有用于用户输出的)

---

## ✅ 验收标准

1. **语法正确**: 
   ```bash
   uv run python -m py_compile scripts/so101.py
   ```

2. **导入正确**:
   ```bash
   uv run python -c "from scripts.so101 import SO101; print('OK')"
   ```

3. **debug 消息不再默认输出** (除非配置 DEBUG 级别)

---

## 📁 相关文件路径

- `/home/admin/Desktop/eai-final-project/scripts/so101.py`

---

// turbo-all
