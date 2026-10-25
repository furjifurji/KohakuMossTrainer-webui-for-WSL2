# WSL2 训练器通用轻量 WebUI 架构与开发指南

> 本文档总结了为 **KohakuMossTrainer** 构建轻量化、现代化 WebUI 的完整设计与实现经验。  
> 按照本指南的架构范式，未来你可以将**任何开源 AI 训练器（如 sd-scripts、Anima、OneTrainer、Flux 等）**放入 WSL2 中，在几十分钟内为其快速搭建一套高颜值、零外部依赖、稳定可靠的专属 Web 控制台。

---

## 目录
1. [核心架构全景：前后端与跨系统解耦](#1-核心架构全景前后端与跨系统解耦)
2. [关键避坑：WSL2 与 Windows 网络通信底线](#2-关键避坑wsl2-与-windows-网络通信底线)
3. [后端设计：Python 标准库零依赖服务](#3-后端设计python-标准库零依赖服务)
4. [Schema 驱动：动态表单与双语提示体系](#4-schema-驱动动态表单与双语提示体系)
5. [配置生成与自适应适配层](#5-配置生成与自适应适配层)
6. [训练进程管理与高并发环形日志流](#6-训练进程管理与高并发环形日志流)
7. [前端工程：现代原生 CSS 变量与深浅主题切换](#7-前端工程现代原生-css-变量与深浅主题切换)
8. [一键启动批处理脚本模板](#8-一键启动批处理脚本模板)

---

## 1. 核心架构全景：前后端与跨系统解耦

### 为什么坚决不要使用 Gradio / Streamlit？
很多训练器喜欢使用 Gradio 或 Streamlit，但在 WSL2 搭配特定训练器的生产环境下，它们存在严重弊病：
- **依赖庞大且冲突**：Gradio 依赖 FastAPI、Starlette、Pydantic 等数十个包，极易与训练环境自身的 PyTorch、Torchvision、HuggingFace 依赖发生严重版本踩踏。
- **无法精细控制训练生命周期**：Gradio 内部的事件循环很难优雅处理训练子进程组、强杀残留 CUDA 显存、阶段性流水线执行（例如：探测 -> 缓存预生成 -> 主训练）。
- **页面状态卡顿与黑盒**：参数多达几十上百项时，Gradio 页面重绘极慢，且难以深度定制专业的前端视觉效果。

### 推荐的轻量化架构
```text
┌─────────────────────────────────────────────────────────────┐
│                    Windows 宿主机操作系统                    │
│                                                             │
│  [ Chrome / Edge 浏览器 ] <---- HTTP (localhost:7878) ───┐ │
│  (原生 HTML5 / CSS3 / Vanilla JS, 零打包工具)            │ │
└──────────────────────────────────────────────────────────┼──┘
                                                           │ (wslrelay 自动端口镜像)
┌──────────────────────────────────────────────────────────┼──┐
│                   WSL2 (Ubuntu Linux) 子系统             ▼  │
│                                                             │
│  [ WebUI 守护进程 ] (webui.py - 仅使用 Python 标准库 http)   │
│         │                                                   │
│         ├──> 提供 API (/api/schema, /api/status, /api/logs) │
│         └──> 任务执行器 (Runner) ────────┐                  │
│                                          ▼                  │
│  [ 训练环境 ] (.venv 虚拟环境，含 PyTorch、CUDA、训练库)     │
│         ├── 阶段 1: 数据探测与分桶统计 (probe)              │
│         ├── 阶段 2: 离线编码预提取 (cache_text / latents)  │
│         └── 阶段 3: 核心多卡/单卡训练 (主训练脚本)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 关键避坑：WSL2 与 Windows 网络通信底线

这是在 WSL2 下开发 WebUI 最容易导致“无法访问此页面（ERR_CONNECTION_REFUSED）”的底层系统机制。

### 坑点 1：Windows Hyper-V 动态端口排除范围（ExcludedPortRange）
- **现象**：在 WSL 里用 `python -m http.server 7847 --bind 0.0.0.0` 启动服务，WSL 本地访问完全正常，Windows 用 WSL IP（如 `172.x.x.x:7847`）也能连，但访问 `http://localhost:7847` 或 `http://127.0.0.1:7847` **百分之百连接拒绝**。
- **原因**：Windows 系统的 Hyper-V / NAT 服务会随机动态圈定多个保留端口段（每次重启都会生成）。
  在 Windows 执行：
  ```powershell
  netsh interface ipv4 show excludedportrange protocol=tcp
  ```
  如果某个端口落在如下范围内（例如 `7756 - 7855` 包含了 `7847`）：
  ```text
  Start Port    End Port
  ----------    --------
        7756        7855    <-- 此时该端口被 Windows 内核全局独占锁定！
  ```
  Windows 的 WSL 端口镜像服务（`wslrelay.exe`）在宿主机创建 `127.0.0.1:7847` 监听时会直接报 `WSAEACCES 10013`（权限被拒绝），导致 Windows 浏览器根本打不开 localhost！
- **解决原则**：
  1. 避免使用常见冲突段（如 `7860` 容易和 ComfyUI/SD-WebUI 冲突，`7890` 容易和代理软件冲突）。
  2. 优先选在 `7856 - 7940` 之间的端口（如 **`7878`**），或者在脚本中允许参数动态指定。

### 坑点 2：控制台被 HTTP 轮询日志无限刷屏
- **现象**：Web 界面通过前端 `setInterval` 每秒轮询 `/api/status` 与 `/api/logs`，控制台每秒疯狂输出几百行 `GET /api/status HTTP/1.1 200 -`，导致真正的训练报错完全看不见。
- **解决**：重写 `BaseHTTPRequestHandler` 的 `log_message` 方法，拦截静默路由：
  ```python
  QUIET_PATHS = ("/api/status", "/api/logs")

  class Handler(BaseHTTPRequestHandler):
      def log_message(self, fmt, *args):
          path = urlparse(self.path).path
          status = str(args[1]) if len(args) > 1 else ""
          if path in QUIET_PATHS or (status.startswith("2") and path.startswith("/static")):
              return  # 彻底屏蔽轮询心跳请求的控制台日志
          # 仅保留真实业务请求和错误日志
          print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")
  ```

### 坑点 3：Windows 与 WSL 路径转换
UI 在 Windows 浏览器中运行，用户习惯复制 `D:\datasets\my_lora`，而 WSL 训练脚本只认 `/mnt/d/datasets/my_lora`：
```python
def to_linux_path(p: str) -> str:
    """Windows 路径 F:\a\b 转换为 WSL Linux 路径 /mnt/f/a/b"""
    p = p.strip().strip('"').strip("'")
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")

def to_windows_path(p: str) -> str:
    """WSL 路径 /mnt/f/a/b 转换为 Windows 显示路径 F:\a\b"""
    if p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:\\" + p[7:].replace("/", "\\")
    return p
```

---

## 3. 后端设计：Python 标准库零依赖服务

在 `webui.py` 中，**完全不需要安装 Flask、FastAPI、Uvicorn**，直接使用 Python 自带的 `http.server.ThreadingHTTPServer`。

### 核心 REST API 路由规范：
- `GET /`：返回 `index.html`
- `GET /static/<file>`：静态资源托管（CSS、JS、SVG 等）
- `GET /api/schema`：返回全量 UI 结构（Tabs、Sections、Fields、默认值、提示词）
- `GET /api/status`：返回任务运行状态（是否运行中、当前阶段、运行时长）
- `GET /api/logs?since=<offset>`：增量获取终端日志
- `POST /api/train`：接收前端 JSON 参数，提交后台训练任务
- `POST /api/stop`：强制终止训练任务进程树
- `POST /api/probe`：仅触发数据集扫描与分桶分析
- `POST /api/preview`：预览最终生成的训练配置文件
- `GET/POST /api/presets`：保存/读取用户自建的任务预设

---

## 4. Schema 驱动：动态表单与双语提示体系

不要在 HTML 里手写每一个 input！  
将全部配置项集中在 `webui_app/schema.py`，由前端动态渲染。这样日后增加或修改参数只需要改几行声明。

### 推荐的字段定义器（Field Builder）
```python
def field(key, label, zh, tip, type_="text", default="", **extra):
    return {"key": key, "label": label, "zh": zh, "tip": tip, "type": type_, "default": default, **extra}

def num(key, label, zh, tip, default, min_=0, max_=999999, step=1, **extra):
    return field(key, label, zh, tip, "number", default, min=min_, max=max_, step=step, **extra)

def flag(key, label, zh, tip, default=False, **extra):
    return field(key, label, zh, tip, "bool", default, **extra)

def sel(key, label, zh, tip, default, options, **extra):
    return field(key, label, zh, tip, "select", default, options=options, **extra)
```

### 极佳的交互特性实现：
1. **中英双语 + 悬停气泡**：
   每一个字段同时赋予英文标识（`label`）、中文简称（`zh`）和详细通俗的原理解释（`tip`）。在 DOM 上挂载 `data-tip` 属性，通过统一的绝对定位悬停气泡显示，杜绝浏览器默认 `title` 迟钝的问题。
2. **条件联动显示（Dynamic Show）**：
   在字段上添加 `show={"bucket_enable": [True]}`，前端统一监听 `onChange`，当父选项不满足时自动添加 `.hidden` 类，避免界面被不相关的复杂参数填满。

---

## 5. 配置生成与自适应适配层

在 `webui_app/configgen.py` 中，编写适配器函数：
输入前端传入的字典 `params`，输出目标训练器所需要的格式（如 Python 变量脚本、YAML 或命令行参数）。

### 经典算法适配经验：
1. **Epoch 到 Step 的换算**：
   许多前沿训练器原生只接受总步数（`STEPS`）和保存间隔步数（`SAVE_EVERY`）。
   - **解决办法**：在正式开始训练前，自动先跑一次极轻量的 `probe`（扫描器），测算出当前数据集每 1 个 Epoch 实际包含多少个 Batch。
   - 然后计算：`STEPS = max_epochs * steps_per_epoch`，`SAVE_EVERY = save_every_epochs * steps_per_epoch`。
2. **动态 Token 与分辨率换算**：
   如果用户选了传统 Batch Size 模式，动态估算其对应的 Max Tokens 预算：
   ```python
   max_tokens = int(batch_size * ((resolution / 16) ** 2) + 1024)
   ```

---

## 6. 训练进程管理与高并发环形日志流

### 进程树优雅终止（Process Group Killing）
在 Linux/WSL 环境下，PyTorch 训练通常会派生多个 DataLoader Worker 进程、DDP 进程或编译进程。如果只调用 `proc.terminate()`，主进程退出后子进程会变成**孤儿进程**继续霸占显卡显存！

**标准的进程树管理范式**：
```python
import os, signal, subprocess

class Runner:
    def _run_subprocess(self, cmd, cwd):
        # 核心：使用 preexec_fn=os.setsid 为当前子进程创建独立的会话/进程组
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid  # 极为关键！
        )
        # 读取输出...

    def stop(self):
        if self.proc and self.proc.poll() is None:
            # 向整个进程组广播 SIGTERM 信号，干净强杀所有子进程
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
```

### 内存友好的环形日志缓冲区（Ring Buffer）
为了防止训练数天产生数万行日志导致内存泄漏或前端卡死，后端维护一个定长滚动日志缓冲队列：
```python
class RingLog:
    def __init__(self, max_lines=4000):
        self.max_lines = max_lines
        self.lines = []
        self.offset = 0  # 全局绝对行索引

    def write(self, line: str):
        self.lines.append(line)
        if len(self.lines) > self.max_lines * 1.5:
            drop = len(self.lines) - self.max_lines
            self.lines = self.lines[drop:]
            self.offset += drop

    def read_since(self, client_offset: int):
        # 客户端按递增 offset 请求增量日志，兼顾零冗余与重连自愈
        if client_offset < self.offset:
            return {"reset": True, "lines": self.lines, "next": self.offset + len(self.lines)}
        idx = client_offset - self.offset
        return {"reset": False, "lines": self.lines[idx:], "next": self.offset + len(self.lines)}
```

---

## 7. 前端工程：现代原生 CSS 变量与深浅主题切换

无需任何前端编译构建工具（Vite / Webpack），仅用纯 CSS 变量实现专业级现代界面。

### 现代化深浅主题调色盘（`style.css`）
```css
:root, [data-theme="dark"] {
  --bg: #0d1117;           /* 背景底色 */
  --bg2: #161b22;          /* 卡片与侧边栏 */
  --bg3: #21262d;          /* 次级背景/悬停 */
  --border: #30363d;       /* 边框分割线 */
  --text: #e6edf3;         /* 主文字 */
  --text2: #8b949e;        /* 副文字 */
  --muted: #6e7681;        /* 弱化提示 */
  --accent: #58a6ff;       /* 品牌重点色 */
  --console-bg: #010409;   /* 终端深黑背景 */
  --console-text: #7ee787; /* 终端黑客绿文字 */
  --shadow: 0 8px 24px rgba(0,0,0,.5);
}

[data-theme="light"] {
  --bg: #f6f8fa;           /* GitHub 经典浅灰底 */
  --bg2: #ffffff;          /* 纯白卡片 */
  --bg3: #eaeef2;          /* 浅灰悬停 */
  --border: #d0d7de;       /* 浅色精致边框 */
  --text: #1f2328;         /* 深度高可读性文字 */
  --text2: #57606a;        /* 次级灰色文字 */
  --muted: #6e7781;        /* 辅助说明 */
  --accent: #0969da;       /* 经典高对比蓝 */
  --console-bg: #1b1f24;   /* 浅色模式下终端仍保持易读深灰底 */
  --console-text: #56d364; /* 清晰终端字色 */
  --shadow: 0 8px 24px rgba(140,149,159,.25);
}

body {
  background: var(--bg);
  color: var(--text);
  transition: background-color .2s ease, color .2s ease;
}
```

### 主题切换逻辑（`app.js`）
```javascript
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("trainer_theme", theme);
  const btn = document.getElementById("btn-theme");
  if (btn) {
    btn.textContent = theme === "light" ? "🌙 深色主题" : "☀️ 浅色主题";
  }
}

function initTheme() {
  const saved = localStorage.getItem("trainer_theme") || "light";
  applyTheme(saved);
  document.getElementById("btn-theme")?.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(cur === "light" ? "dark" : "light");
  });
}
```

---

## 8. 一键启动批处理脚本模板

在项目根目录下创建 `启动训练器(网页端).bat`：
```bat
@echo off
chcp 65001 >nul
title AI Trainer Studio (WSL2 WebUI)
cd /d "%~dp0"

echo ============================================================
echo   正在启动训练器控制台 (WSL2)...
echo   访问地址: http://localhost:7878
echo   关闭本窗口即可彻底停止服务。
echo ============================================================
echo.

rem 3 秒后自动拉起默认浏览器
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:7878"

rem 调用 WSL 执行虚拟环境中的 WebUI 脚本
wsl -d Ubuntu -- bash -c ".venv/bin/python webui.py 7878"

pause
```

---

## 总结 CheckList（复用检查清单）

当你要给下一个新训练器套上这套 WebUI 时，只需执行以下 5 步：
1. **复制模板**：将 `webui.py`、`webui_app/` 以及 `.bat` 启动脚本复制到新训练器根目录。
2. **改 `schema.py`**：根据新训练器的参数要求，把各个超参数填入 `fields`。
3. **改 `configgen.py`**：将前端收集到的 `params` 转换为新训练器读取的配置文件或 CLI 启动参数。
4. **绑定 `runner.py`**：将 `runner` 中的执行命令换成新训练器的入口命令（如 `python train.py ...`）。
5. **双击 `.bat` 运行**：浏览器自动打开，立刻享有双语提示、深浅色切换、分桶探测与可视化任务控制！
