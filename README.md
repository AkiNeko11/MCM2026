this is a repo for MCM2026 C

## Quick Start

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/AkiNeko11/MCM2026.git
cd MCM2026

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 安装依赖
# 默认直接执行安装的是cpu版本的torch，如需使用gpu版本的请先安装torch再执行
pip install -r requirements.txt
```