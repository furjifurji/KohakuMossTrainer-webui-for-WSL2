"""UI schema of the KohakuMoss WebUI: tabs -> sections -> fields.

Single source of truth for the form (rendered by static/app.js), the defaults, the
Chinese hover tips and the validation hints. `configgen.py` maps the resulting
parameter dict onto KohakuMoss knobs (scripts/train/moss.py globals).

Field keys: key, label (English), zh (Chinese name), tip (Chinese hover text), type
(text|path|number|select|bool|textarea|datasets), default, options, min/max/step,
show (dict: other-field -> value or list of values), placeholder, kind (dir|file),
wide (full row).
"""


def opt(value, label, tip=""):
    return {"value": value, "label": label, "tip": tip}


def field(key, label, zh, tip, type="text", default="", **kw):
    return {"key": key, "label": label, "zh": zh, "tip": tip, "type": type,
            "default": default, **kw}


def num(key, label, zh, tip, default, min=None, max=None, step=None, **kw):
    return field(key, label, zh, tip, "number", default, min=min, max=max, step=step, **kw)


def sel(key, label, zh, tip, default, options, **kw):
    return field(key, label, zh, tip, "select", default, options=options, **kw)


def flag(key, label, zh, tip, default=False, **kw):
    return field(key, label, zh, tip, "bool", default, **kw)


def path(key, label, zh, tip, default="", kind="file", **kw):
    return field(key, label, zh, tip, "path", default, kind=kind, **kw)


DEFAULT_COMFY = r"F:\AIself\A_Comfyui\ComfyUI-aki-v3\ComfyUI\models"
DEFAULT_DIT = DEFAULT_COMFY + r"\diffusion_models\moss\diffusion_models_moss_image_base_bf16.safetensors"
DEFAULT_TE = DEFAULT_COMFY + r"\text_encoders\moss\qwen3.5_9b_int8_convrot.safetensors"
DEFAULT_VAE = DEFAULT_COMFY + r"\vae\moss\flux2-vae_small.safetensors"

PRECISION_OPTS = [
    opt("auto", "auto", "自动：保持权重文件原本的精度（bf16 文件即 16 位全精度，int8 文件即 int8 量化）。"),
    opt("compute", "compute", "以 16 位（bf16/fp16）全精度计算；若文件是量化文件会先反量化。"),
    opt("int8", "int8", "把底模量化为 int8 以节省显存，质量略有损失。"),
    opt("fp8", "fp8", "fp8 仅权重量化（需要新架构显卡）。"),
    opt("fp8_compute", "fp8_compute", "fp8 矩阵乘法加速（sm89+，RTX 40 系可用）。"),
    opt("mxfp8", "mxfp8", "MXFP8 微缩放 fp8 计算（sm89+）。"),
]

TABS = [
    # ------------------------------------------------------------------ Training
    {"id": "training", "label": "Training", "zh": "训练", "tip": "批次、梯度累加、训练轮数、优化器、学习率与输出保存设置。", "sections": [
        {"title": "Batch & Duration", "zh": "批次与时长", "tip": "每步样本数、梯度累加以及训练总时长（按 epoch 或步数）。", "fields": [
            num("batch_size", "Batch Size", "批大小", "每次前向/反向处理的图片张数（仅在批处理方式为 count 时生效）。16GB 显存建议 1~2，大分辨率请保持 1。", 1, 1, 64, 1),
            num("grad_acc", "Gradient Accumulation", "梯度累加步数", "累计多少个批次后才更新一次权重。有效批量 = 批大小 × 梯度累加。显存不够时用它放大有效批量，不额外占显存。", 4, 1, 256, 1),
            sel("duration_unit", "Duration Unit", "训练时长单位", "选择按 epoch（遍历整个数据集的次数）还是按优化器步数来决定训练多久。", "epochs", [
                opt("epochs", "Epochs", "按 epoch 训练与保存。每个 epoch 的步数会根据数据集自动换算。"),
                opt("steps", "Steps", "按优化器步数训练与保存。"),
            ]),
            num("max_epochs", "Max Epochs", "最大训练轮数", "训练的 epoch 总数（一个 epoch = 数据集中每张图按其重复次数遍历一遍）。", 20, 1, 10000, 1, show={"duration_unit": "epochs"}),
            num("save_lora_every_epochs", "Export LoRA Every N Epochs", "每 N 个 epoch 导出 LoRA", "每隔多少个 epoch 导出一次轻量 LoRA 文件（用于画图测试，仅约百兆，存入 [输出目录]/[任务名]/loras/ 文件夹）。默认 1。", 1, 1, 1000, 1, show={"duration_unit": "epochs"}),
            num("save_every_epochs", "Save Checkpoint Every N Epochs", "每 N 个 epoch 保存续训权重", "每隔多少个 epoch 保存一次完整的续训检查点（包含优化器 optim.pt + 动量矩，存入 [输出目录]/[任务名]/checkpoints/，用于 Resume 断点续训）。默认 10。", 10, 0, 1000, 1, show={"duration_unit": "epochs"}),
            num("max_steps", "Max Steps", "最大训练步数", "优化器更新的总步数（每步 = 批大小 × 梯度累加 张图）。", 1000, 1, 1000000, 10, show={"duration_unit": "steps"}),
            num("save_lora_every_steps", "Export LoRA Every N Steps", "每 N 步导出 LoRA", "每隔多少步导出一次轻量 LoRA 文件（存入 [输出目录]/[任务名]/loras/ 文件夹）。默认 100。", 100, 1, 1000000, 10, show={"duration_unit": "steps"}),
            num("save_every_steps", "Save Checkpoint Every N Steps", "每 N 步保存续训权重", "每隔多少步保存一次完整的续训检查点（含 optim.pt，存入 [输出目录]/[任务名]/checkpoints/，用于断点续训）。默认 500。", 500, 0, 1000000, 10, show={"duration_unit": "steps"}),
            num("keep_last", "Keep Last N Checkpoints", "仅保留最近 N 个续训检查点", "仅保留最近 N 个带有 optim.pt 的完整检查点目录以节省磁盘；0 表示全部保留。导出的 loras/ 永远不会被删除。", 2, 0, 1000, 1),
        ]},
        {"title": "Optimizer", "zh": "优化器与学习率", "tip": "优化算法、学习率及其调度曲线。", "fields": [
            sel("optimizer", "Optimizer", "优化器", "更新权重所用的算法。", "adamw", [
                opt("adamw", "AdamW", "标准 AdamW（CUDA 上为融合实现），最稳妥的选择。"),
                opt("adamw8bit_kernel", "AdamW8bit (kernel)", "KohakuMoss 自带的 8-bit AdamW，优化器状态显存大幅减少。"),
                opt("adamw8bit", "AdamW8bit (bitsandbytes)", "bitsandbytes 的 8-bit AdamW。"),
                opt("adamw16", "AdamW16", "16 位参数专用 AdamW，状态用 16 位保存。"),
                opt("prodigy", "Prodigy", "自适应学习率优化器，学习率通常填 1.0。"),
                opt("muon", "Muon", "对 2D 隐藏矩阵使用 Muon 的优化器（不能与 low/minimal 的 fused_step 同用）。"),
                opt("schedule_free", "ScheduleFree", "无调度器（Schedule-Free）优化器。"),
            ]),
            field("learning_rate", "Learning Rate", "学习率", "基础学习率，支持科学计数法，如 1e-4。LoRA 常用 1e-4~2e-4；Prodigy 请填 1.0。", "text", "1e-4"),
            sel("lr_scheduler", "LR Scheduler", "学习率调度器", "学习率随训练进度变化的曲线。", "cosine", [
                opt("cosine", "Cosine", "余弦退火：先热身，然后按余弦曲线平滑下降，收敛稳定。"),
                opt("constant", "Constant", "恒定学习率（热身之后保持不变）。"),
                opt("linear", "Linear", "线性衰减至最小学习率。"),
            ]),
            num("lr_warmup_steps", "LR Warmup Steps", "学习率热身步数", "训练最初若干优化器步内学习率从 0 线性升到设定值，防止早期震荡。", 50, 0, 100000, 10),
            num("lr_min_ratio", "Min LR Ratio", "最小学习率比例", "学习率衰减的下限，占初始学习率的比例（例如 0.1 表示衰减到 10%）。", 0.0, 0, 1, 0.01, show={"lr_scheduler": ["cosine", "linear"]}),
            num("weight_decay", "Weight Decay", "权重衰减", "权重衰减系数，用于抑制过拟合。", 0.01, 0, 1, 0.001),
            num("grad_clip", "Gradient Clip", "梯度裁剪", "梯度全局范数的裁剪阈值，防止梯度爆炸。0 表示不裁剪。", 1.0, 0, 100, 0.1),
            num("beta1", "Adam Beta1", "Adam Beta1", "Adam 一阶动量系数。", 0.9, 0, 1, 0.01),
            num("beta2", "Adam Beta2", "Adam Beta2", "Adam 二阶动量系数（MOSS 默认 0.95）。", 0.95, 0, 1, 0.001),
            num("ema", "EMA Decay", "EMA 衰减率", "权重指数滑动平均衰减率，0 为关闭，常用 0.9999。", 0.0, 0, 1, 0.0001),
        ]},
        {"title": "Output", "zh": "输出与保存", "tip": "保存位置、文件名与格式。", "fields": [
            path("output_dir", "Output Directory", "输出目录", "训练结果保存目录，可填 Windows 路径（如 F:\\xxx），会自动映射到 WSL。", r"F:\AIself\A_Lora_Train\output", kind="dir", wide=True),
            field("run_name", "Run Name", "任务名称", "本次训练的名称，会作为输出子目录和 LoRA 文件名前缀。", "text", "my_moss_lora"),
            sel("save_format", "Save Format", "保存格式", "导出的 LoRA 文件格式。", "comfy", [
                opt("comfy", "ComfyUI", "ComfyUI 可直接加载的 LoRA（推荐）。"),
                opt("hf", "HF / LyCORIS", "带 diffusers 键名的 LyCORIS 文件。"),
                opt("both", "Both", "两种格式都导出。"),
                opt("none", "None", "不导出（仅保留检查点）。"),
            ]),
            flag("save_merged", "Save Merged Model", "同时保存合并底模", "训练结束时同时导出把 LoRA 合并进底模后的完整模型（体积很大）。"),
            field("resume", "Resume From", "继续训练自 (断点续训)", "续训检查点路径。填 latest 则自动从输出目录中寻找最新的 step_XXXXXXXX 检查点接续训练；留空表示全新训练。", "text", "", placeholder="填 latest 或 step_XXXXXXXX 目录路径"),
            num("seed", "Seed", "随机种子", "固定种子便于复现和对比。", 42, 0, 2147483647, 1),
        ]},
    ]},
    # ------------------------------------------------------------------ Model
    {"id": "model", "label": "Model", "zh": "模型", "tip": "DiT 底模、文本编码器与 VAE 的路径与精度。", "sections": [
        {"title": "Weights", "zh": "权重文件", "tip": "底模三件套：DiT（扩散主干）、Text Encoder（Qwen3.5 文本编码器）、VAE。可从下拉列表选择已扫描到的文件。", "fields": [
            path("weights_root", "Weights Root", "权重根目录", "ComfyUI 的 models 目录（含 diffusion_models / text_encoders / vae）。下面三个路径留空时在此目录自动查找。", DEFAULT_COMFY, kind="dir", wide=True),
            path("transformer", "DiT / Transformer", "DiT 底模", "MOSS-Image DiT 扩散主干权重 .safetensors。bf16 为原精度（推荐，质量最好）；int8_convrot 为量化版（显存约一半，类似 QLoRA，质量略低）。", DEFAULT_DIT, kind="file", wide=True, scan="diffusion_models"),
            path("text_encoder", "Text Encoder", "文本编码器", "Qwen3.5-9B 文本编码器 .safetensors（bf16 / int8_convrot / w4a8）。仅用于生成文本缓存，int8 版对结果影响很小且更省内存。", DEFAULT_TE, kind="file", wide=True, scan="text_encoders"),
            path("vae", "VAE", "VAE", "FLUX.2 VAE .safetensors（flux2-vae_small 为 MOSS 官方使用的版本）。", DEFAULT_VAE, kind="file", wide=True, scan="vae"),
            sel("variant", "Variant", "模型版本", "MOSS-Image 的版本，决定默认查找的 DiT 文件名。", "base", [
                opt("base", "Base", "Base 基础版（训练 LoRA 推荐）。"),
                opt("sft", "SFT", "SFT 指令微调版。"),
            ]),
        ]},
        {"title": "Precision & Placement", "zh": "精度与放置", "tip": "计算精度以及文本编码器/VAE 放在哪里运行。", "fields": [
            sel("precision", "Compute Precision", "计算精度", "训练所用的计算精度。RTX 40 系推荐 bf16。", "bf16", [
                opt("bf16", "bf16", "bfloat16：动态范围大，不易溢出（推荐）。"),
                opt("fp16", "fp16", "float16：旧显卡（V100/2080Ti）使用，需要在线残差修正。"),
                opt("auto", "auto", "自动选择：支持 bf16 则用 bf16，否则 fp16。"),
            ]),
            sel("base_precision", "Base Precision", "底模精度", "冻结的底模权重以什么精度参与训练。选 auto 则沿用文件精度：bf16 文件即全精度；int8 文件即 int8 量化训练。", "auto", PRECISION_OPTS),
            sel("text_precision", "Text Encoder Precision", "文本编码器精度", "文本编码器的计算/存储精度。仅用于生成文本缓存，auto 即沿用文件精度。", "auto", PRECISION_OPTS[:3]),
            sel("text_memory", "Text Encoder Memory", "文本编码器显存策略", "文本编码器如何放置在显存中。", "auto", [
                opt("auto", "auto", "自动选择能装下的最快方案。"),
                opt("resident", "resident", "常驻显存（最快，占用最多）。"),
                opt("offload", "offload", "每次推理时整体载入、用完卸载。"),
                opt("ring", "ring", "逐层流式环形缓冲（最省显存）。"),
            ]),
            sel("text_device", "Text Encoder Device", "文本编码器设备", "文本编码器运行的设备。选 cpu 仅在文本缓存为 offline 时可用。", "", [
                opt("", "Training GPU", "与训练同一块 GPU。"),
                opt("cpu", "CPU", "在 CPU 上编码（很慢，但完全不占显存）。"),
            ]),
            sel("vae_dtype", "VAE Dtype", "VAE 数据类型", "VAE 编码/解码使用的数据类型。", "auto", [
                opt("auto", "auto", "自动选择（支持则 bf16，否则 fp16）。"),
                opt("bf16", "bf16", "bfloat16。"),
                opt("fp16", "fp16", "float16。"),
                opt("fp32", "fp32", "float32（最精确，占用更多）。"),
            ]),
            sel("vae_memory", "VAE Memory", "VAE 显存策略", "VAE 如何放置在显存中。", "auto", [
                opt("auto", "auto", "自动选择。"),
                opt("resident", "resident", "常驻显存。"),
                opt("offload", "offload", "用时载入、用完卸载。"),
                opt("ring", "ring", "分块环形流式。"),
                opt("cpu", "cpu", "在 CPU 上编码（仅 latent 缓存为 offline 时）。"),
            ]),
            flag("vae_tile", "VAE Tiling", "VAE 分块编码", "分块（tiled）编码/解码，降低超大图片的 VAE 显存峰值，速度略慢。"),
        ]},
    ]},
    # ------------------------------------------------------------------ Dataset
    {"id": "dataset", "label": "Dataset", "zh": "数据集", "tip": "训练图片目录、分辨率、分桶、标注与缓存设置。", "sections": [
        {"title": "Dataset Folders", "zh": "数据集目录", "tip": "可添加多个数据集目录，每个目录可单独指定重复倍率。", "fields": [
            field("datasets", "Dataset Folders", "数据集目录列表", "训练图片目录列表。kohya 结构要求子文件夹形如 10_名称（10 为重复次数）；folder 结构直接读取目录内的图片+同名 txt。", "datasets", [
                {"path": r"F:\AIself\A_Lora_Train\train_data", "source": "kohya", "weight": 1}
            ], wide=True),
        ]},
        {"title": "Resolution & Bucketing", "zh": "分辨率与分桶", "tip": "训练分辨率、宽高比分桶以及图片尺寸范围。", "fields": [
            num("resolution", "Resolution", "训练分辨率", "训练目标分辨率（边长，像素面积 = 边长²）。可设为 1536 等。分桶时每个桶的面积约等于该值的平方。", 1024, 256, 4096, 64),
            flag("bucket_enable", "Enable Bucketing", "启用分桶训练", "按图片宽高比分组，每批只含同尺寸图片，避免裁剪过多。默认启用。", True),
            sel("non_bucket_mode", "Non-bucket Mode", "非分桶模式", "未启用分桶时的尺寸处理方式。", "resize", [
                opt("resize", "Resize", "保持原始宽高比缩放到目标面积。"),
                opt("square_crop", "Square Crop", "居中裁剪为正方形。"),
                opt("shifted_square", "Shifted Square", "随机位置裁剪正方形（不能用于 offline latent 缓存）。"),
            ], show={"bucket_enable": [False]}),
            num("min_bucket_reso", "Min Bucket Resolution", "最小桶边长", "允许的图片/桶最短边（像素）。用于限制极端长宽比：最短边不会小于该值。", 256, 64, 4096, 16, show={"bucket_enable": [True]}),
            num("max_bucket_reso", "Max Bucket Resolution", "最大桶边长", "允许的图片/桶最长边（像素）。用于限制极端长宽比：最长边不会超过该值。", 3072, 256, 8192, 16, show={"bucket_enable": [True]}),
            num("bucket_reso_steps", "Bucket Resolution Steps", "桶分辨率步长", "桶尺寸的步进（像素），必须是 16 的倍数；越小桶越多越精细，常用 32 或 64。", 32, 16, 256, 16),
            flag("no_upscale", "Do Not Upscale", "禁止放大小图", "小于桶分辨率的图片不会被强行放大：直接使用其原始尺寸（取步长整数倍）的独立小桶；大于目标面积的图片则会等比缩小。", True),
            sel("crop", "Crop Mode", "裁剪方式", "裁剪到桶宽高比时的裁剪位置。随机裁剪不能配合 offline latent 缓存。", "center", [
                opt("center", "Center", "居中裁剪。"),
                opt("random", "Random", "随机位置裁剪（需 latent 缓存非 offline）。"),
            ]),
            flag("flip", "Horizontal Flip", "水平翻转增强", "随机水平翻转图片做数据增强（会使 latent 缓存翻倍）。"),
            sel("background", "Alpha Background", "透明背景填充", "透明 PNG 的透明区域填充色。", "white", [
                opt("white", "White", "白色。"), opt("black", "Black", "黑色。"), opt("gray", "Gray", "灰色。"),
            ]),
            sel("resample", "Resample Filter", "缩放滤镜", "图片缩放时使用的重采样算法。", "lanczos", [
                opt("lanczos", "Lanczos", "Lanczos：最清晰（推荐）。"),
                opt("bicubic", "Bicubic", "双三次插值。"),
                opt("bilinear", "Bilinear", "双线性插值。"),
                opt("box", "Box", "盒式滤波。"),
            ]),
        ]},
        {"title": "Batching", "zh": "批处理方式", "tip": "如何把图片组成训练批次。", "fields": [
            sel("batching", "Batching", "批处理方式", "批次的组成方式。", "count", [
                opt("count", "Count (Batch Size)", "固定张数：每批 Batch Size 张同尺寸图片（与传统训练器一致）。"),
                opt("tokens", "Tokens", "按 token 预算自动凑批：把多张图拼到不超过 Max Tokens（效率最高）。"),
                opt("tokens_ffd", "Tokens FFD", "按 token 预算装箱（首次适应递减），减少填充浪费。"),
            ]),
            num("max_tokens", "Max Tokens", "最大 Token 数", "仅 tokens 模式：每批最多的 token 数（图像+文本）。count 模式下由程序按 Batch Size × 最大桶 自动估算。16GB 显卡 1024 分辨率可用 8192~12288。", 12288, 1024, 131072, 512, show={"batching": ["tokens", "tokens_ffd"]}),
            sel("overlong", "Over-long Samples", "超长样本处理", "单个样本总 token（图像+文本）超过上限时的处理。1024分辨率下默认上限为 5120 tokens（4096图像+1024文本预留），普通样本绝不会超限。", "drop_tokens", [
                opt("drop", "Drop", "丢弃该样本。"),
                opt("truncate_text", "Truncate Text", "截断文本。"),
                opt("drop_tokens", "Drop Image Tokens", "随机丢弃部分图像 token 使其放得下（推荐）。"),
            ]),
            flag("drop_last", "Drop Last Partial Batch", "丢弃最后不足的批次", "count 模式下，丢弃每个桶末尾凑不满 Batch Size 的批次。关闭可避免稀有尺寸的图片被丢弃（推荐关闭）。", False, show={"batching": ["count"]}),
        ]},
        {"title": "Captions", "zh": "标注", "tip": "标注 txt 的读取与增强方式。", "fields": [
            sel("caption_template", "Caption Template", "标注模板", "如何由标注文件生成提示词。", "tags", [
                opt("tags", "Tags", "逗号分隔的 tag 标注（danbooru 风格）。"),
                opt("nl", "Natural Language", "自然语言描述；没有时回退到 tags。"),
            ]),
            flag("shuffle_caption", "Shuffle Caption", "打乱标注 tag", "每次训练随机打乱 tag 顺序，增强泛化、防止顺序过拟合。", True, show={"caption_template": ["tags"]}),
            num("keep_tokens", "Keep Tokens", "保留前 N 个 tag", "打乱时固定开头的 N 个 tag（如触发词）不参与打乱。", 0, 0, 100, 1, show={"caption_template": ["tags"]}),
            num("tag_dropout", "Tag Dropout", "tag 随机丢弃率", "每个 tag 被随机丢弃的概率。0 代表 0%，0.1 代表 10%（范围 0~1，用于增强泛化，通常保持 0 即可）。", 0.0, 0, 1, 0.01, show={"caption_template": ["tags"]}),
            flag("underscores", "Underscores to Spaces", "下划线转空格", "把 tag 中的下划线替换为空格（blue_hair → blue hair）。", True, show={"caption_template": ["tags"]}),
            flag("artist_from_folder", "Folder Name as First Tag", "文件夹名作为首个标签", "kohya 结构下，把 N_名称 文件夹的“名称”作为每张图的首个标签（如角色/画师名）。", False),
            field("caption_prefix", "Caption Prefix", "标注前缀（触发词）", "添加到每条标注最前面的固定文本，如触发词。", "text", "", placeholder="如：mychar, "),
            field("caption_suffix", "Caption Suffix", "标注后缀", "添加到每条标注末尾的固定文本。", "text", ""),
            num("text_drop", "Caption Dropout", "标注丢弃概率", "整条标注被替换为空提示词的概率。0.1 代表 10%（范围 0~1，0=不丢弃，0.1=10%，用于 CFG 无条件引导训练，推荐 0.1）。", 0.1, 0, 1, 0.01),
            num("caption_variants", "Caption Variants", "标注变体数", "离线文本缓存时每张图预先生成的随机打乱/丢弃变体数（默认 4 种）。训练时每轮随机选一种，既不用将 9B 文本编码器常驻显存，又能保留 tag 随机打乱的多样性。", 4, 1, 64, 1),
        ]},
        {"title": "Caching", "zh": "缓存与加载", "tip": "VAE 潜变量与文本嵌入缓存、数据加载线程。", "fields": [
            sel("latent_cache", "Latent Cache", "Latent 缓存", "VAE 潜变量缓存方式。offline 会在训练前先预编码全部图片，训练时完全不需要加载 VAE。", "offline", [
                opt("offline", "Offline", "离线预缓存（训练前自动运行缓存步骤，最省显存最快）。"),
                opt("online", "Online", "训练时边编码边缓存。"),
                opt("off", "Off", "不缓存，每步现场编码。"),
            ]),
            sel("text_cache", "Text Cache", "文本缓存", "文本编码缓存方式。offline 预先将标注用 9B 编码器转为嵌入（配合 Caption Variants 仍可享受 tag 随机打乱），训练时无需加载 9B 编码器（最省显存）。", "offline", [
                opt("offline", "Offline", "离线预缓存（推荐，配合变体数仍有随机打乱）。"),
                opt("online", "Online", "训练时边编码边缓存。"),
                opt("off", "Off", "不缓存，每步现场编码（需额外显存）。"),
            ]),
            path("cache_dir", "Cache Directory", "缓存目录", "缓存存放目录；留空则放在数据集目录下的 cache 文件夹。", "", kind="dir", wide=True),
            sel("cache_dtype", "Cache Dtype", "缓存精度", "缓存数据的存储精度。", "bf16", [
                opt("bf16", "bf16", "bfloat16。"), opt("fp16", "fp16", "float16。"),
                opt("fp32", "fp32", "float32（体积最大）。"), opt("fp8", "fp8", "fp8（体积最小）。"),
            ]),
            num("num_workers", "DataLoader Workers", "数据加载线程数", "数据加载子进程数量。", 4, 0, 32, 1),
            num("prefetch", "Prefetch", "预取批次数", "每个加载进程预取的批次数。", 2, 1, 16, 1),
        ]},
    ]},
    # ------------------------------------------------------------------ Network
    {"id": "network", "label": "Network", "zh": "网络", "tip": "LoRA/LyCORIS 算法、秩、Alpha 及注入层。", "sections": [
        {"title": "Adapter", "zh": "适配器结构", "tip": "微调算法与其核心超参数。", "fields": [
            sel("algo", "Algorithm", "微调算法", "微调所用的 LyCORIS 算法。", "lora", [
                opt("lora", "LoRA", "标准 LoRA，最稳妥通用。"),
                opt("locon", "LoCon", "对卷积层也适配的 LoRA。"),
                opt("loha", "LoHa", "哈达玛积低秩分解，表现力更强。"),
                opt("lokr", "LoKr", "克罗内克积分解，参数效率最高。"),
                opt("oft", "OFT", "正交微调。"),
                opt("boft", "BOFT", "蝶形正交微调。"),
                opt("glora", "GLoRA", "广义 LoRA。"),
                opt("ia3", "IA3", "极轻量的缩放向量适配。"),
                opt("dylora", "DyLoRA", "动态秩 LoRA。"),
                opt("tlora", "T-LoRA", "时间步相关 LoRA。"),
                opt("full", "Full Finetune", "全参数微调（显存需求极高，16GB 显卡不推荐）。"),
            ]),
            num("dim", "Network Dim (Rank)", "网络维度 Rank", "LoRA 的秩。越大容量越强、文件越大；常用 16~64。对 LoKr 无效。", 32, 1, 1024, 1, show={"algo": ["lora", "locon", "loha", "dylora", "tlora", "oft", "boft", "glora", "ia3"]}),
            num("alpha", "Network Alpha", "网络 Alpha", "权重缩放系数，实际缩放 = Alpha / Rank。常设为 Rank 的一半或与 Rank 相同。", 16, 0.1, 1024, 0.5, show={"algo": ["lora", "locon", "loha", "dylora", "tlora", "oft", "boft", "glora", "ia3"]}),
            num("lokr_factor", "LoKr Factor", "LoKr 分解因子", "LoKr 的矩阵分解因子（常用 4/8/16，数值越小越接近全矩阵）。", 8, 1, 128, 1, show={"algo": ["lokr"]}),
            flag("lokr_full_matrix", "LoKr Full Matrix", "LoKr 完整矩阵", "LoKr 的第二个因子使用完整矩阵。", False, show={"algo": ["lokr"]}),
            flag("lokr_decompose_both", "LoKr Decompose Both", "LoKr 双向分解", "同时分解 LoKr 的两个因子。", False, show={"algo": ["lokr"]}),
            sel("preset", "Target Preset", "注入层预设", "LoRA 注入到哪些层。", "attn-mlp", [
                opt("attn", "Attention", "只注入注意力层，最轻量。"),
                opt("attn-mlp", "Attention + MLP", "注意力 + 前馈层（推荐）。"),
                opt("full-block", "Full Block", "整个 Transformer 块的全部线性层。"),
                opt("all", "All", "所有可适配层。"),
            ]),
            flag("preset_text_refiner", "+ Text Refiner", "加上文本精炼块", "额外对 2 个文本精炼块注入适配器。", False),
            flag("preset_modulation", "+ Modulation", "加上调制层", "额外对 AdaLN 调制层注入适配器。", False),
            flag("preset_embed", "+ Embedding", "加上嵌入层", "额外对嵌入层注入适配器。", False),
            sel("adapter_mode", "Adapter Mode", "适配器模式", "适配器在前向中的计算方式。", "bypass", [
                opt("bypass", "Bypass", "旁路计算（兼容量化底模，推荐）。"),
                opt("merge", "Merge", "每步合并进权重。"),
            ]),
        ]},
        {"title": "Regularization", "zh": "正则与变体", "tip": "Dropout、DoRA、LoRA+ 等增强设置。", "fields": [
            flag("dora", "DoRA", "启用 DoRA", "权重分解 LoRA，细节更好，显存与耗时略增。", False, show={"algo": ["lora", "locon"]}),
            flag("dora_detach", "DoRA Detach Norm", "DoRA 分离范数", "训练中分离 DoRA 的权重范数梯度（更省显存）。", True, show={"dora": [True]}),
            num("dropout_input", "Input Dropout", "输入 Dropout", "适配器输入的 Dropout 概率。", 0.0, 0, 1, 0.01),
            num("dropout_rank", "Rank Dropout", "Rank Dropout", "随机丢弃秩维度的概率，降低过拟合。", 0.0, 0, 1, 0.01),
            num("dropout_module", "Module Dropout", "模块 Dropout", "整个适配器模块被随机跳过的概率。", 0.0, 0, 1, 0.01),
            num("loraplus_ratio", "LoRA+ Ratio", "LoRA+ 学习率比", "LoRA+：上投影矩阵相对下投影的学习率倍率，1.0 为关闭，可尝试 4~16。", 1.0, 1, 128, 0.5),
            flag("train_norms", "Train Norms", "同时训练 Norm", "同时训练归一化层参数。", False),
            flag("use_tucker", "Use Tucker", "使用 Tucker 分解", "对卷积类算法使用 Tucker 分解。", False),
            flag("use_scalar", "Use Scalar", "使用标量缩放", "为适配器引入可学习的标量缩放。", False),
            flag("rank_dropout_scale", "Rank Dropout Scale", "Rank Dropout 缩放", "Rank Dropout 时对输出重新缩放。", False),
        ]},
    ]},
    # ------------------------------------------------------------------ Samples
    {"id": "samples", "label": "Samples", "zh": "采样", "tip": "训练过程中的测试出图设置。", "sections": [
        {"title": "In-Training Sampling", "zh": "训练中出图", "tip": "每隔一段时间用当前 LoRA 生成测试图，查看学习效果。", "fields": [
            flag("sample_enable", "Enable Sampling", "启用训练中出图", "开启后会定期生成测试图（会额外占用时间与显存）。", False),
            num("sample_every_epochs", "Sample Every N Epochs", "每 N 个 epoch 出图", "每隔多少个 epoch 出一次图。", 1, 1, 1000, 1, show={"sample_enable": [True], "duration_unit": "epochs"}),
            num("sample_every_steps", "Sample Every N Steps", "每 N 步出图", "每隔多少个优化器步出一次图。", 100, 1, 1000000, 10, show={"sample_enable": [True], "duration_unit": "steps"}),
            field("sample_prompts", "Sample Prompts", "出图提示词", "测试出图使用的提示词，每行一条。", "textarea", "a lighthouse on a rocky coast at sunset", wide=True, show={"sample_enable": [True]}),
            field("sample_negative", "Negative Prompt", "负面提示词", "测试出图的负面提示词。", "text", "", wide=True, show={"sample_enable": [True]}),
            num("sample_steps", "Sampler Steps", "采样步数", "测试出图的去噪步数。", 30, 1, 200, 1, show={"sample_enable": [True]}),
            num("sample_cfg", "CFG Scale", "CFG 强度", "提示词引导强度。", 3.0, 0, 30, 0.1, show={"sample_enable": [True]}),
            num("sample_shift", "Sampler Shift", "采样 Shift", "采样时间步偏移。", 5.0, 0, 20, 0.1, show={"sample_enable": [True]}),
            num("sample_width", "Width", "出图宽度", "测试图宽度（像素，16 的倍数）。", 1024, 256, 4096, 16, show={"sample_enable": [True]}),
            num("sample_height", "Height", "出图高度", "测试图高度（像素，16 的倍数）。", 1024, 256, 4096, 16, show={"sample_enable": [True]}),
            field("sample_seeds", "Sample Seeds", "出图种子", "测试出图的种子，多个用英文逗号分隔。", "text", "0", show={"sample_enable": [True]}),
        ]},
    ]},
    # ------------------------------------------------------------------ Advanced
    {"id": "advanced", "label": "Advanced", "zh": "高级", "tip": "显存策略、扩散目标函数、内核调优以及自定义配置覆盖。", "sections": [
        {"title": "Memory", "zh": "显存策略", "tip": "决定权重/激活/优化器状态放在显存还是内存。", "fields": [
            sel("memory_profile", "Memory Profile", "显存档案", "显存使用策略。auto 会自动选择“估算能放进显存的最快档案”，16G 显卡推荐优先匹配 ckpt-resident。", "auto", [
                opt("auto", "auto", "自动选择最快且放得下的档案（16G 显卡会自动匹配 ckpt-resident，推荐）。"),
                opt("ckpt-resident", "ckpt-resident (推荐 16G/24G 显卡)", "梯度检查点 + 底模全常驻显存（~12G 显存，WSL2 最稳定且最快，完全不占 CPU 锁页内存）。"),
                opt("speed", "speed", "全部常驻显存，无重计算，最快，需要极大显存（约 35GB+）。"),
                opt("balanced", "balanced", "块检查点 + 16 个常驻块，其余内存流式（注意：WSL2 易报 cuMemHostAlloc 内存错误）。"),
                opt("low", "low", "4 个常驻块 + 激活卸载 + 主机端优化器状态（约 4GB@1MP，更慢但更省）。"),
                opt("minimal", "minimal", "2 个常驻块 + 梯度卸载（极限省显存）。"),
                opt("manual", "manual", "手动：使用下方“手动显存方案”中的参数。"),
            ]),
            flag("memory_auto_force", "Force Last Profile", "强制使用最后档案", "auto 找不到放得下的档案时，强行使用最后一个（带警告）而不是报错。"),
            sel("step_release", "Step Release", "步间释放显存", "每步之间是否释放缓存显存。", "none", [
                opt("none", "none", "不释放（最快）。"),
                opt("empty_cache", "empty_cache", "每步清空缓存（会同步，较慢但缓解碎片）。"),
            ]),
            sel("checkpoint", "Checkpointing", "梯度检查点", "仅 manual 档案：激活重计算粒度。", "block", [
                opt("block", "block", "按块重计算。"), opt("sub-block", "sub-block", "子块级重计算（更省）。"),
                opt("selective", "selective", "选择性重计算。"), opt("none", "none", "不重计算（最快，最占显存）。"),
            ], show={"memory_profile": ["manual"]}),
            sel("stream", "Block Streaming", "块权重流式", "仅 manual 档案：底模块权重是否流式加载。", "ring", [
                opt("off", "off", "全部常驻显存。"), opt("ring", "ring", "冻结底模按环形缓冲流式（LoRA 推荐）。"),
                opt("swap", "swap", "全量微调的主机端主权重交换。"),
            ], show={"memory_profile": ["manual"]}),
            num("stream_resident", "Resident Blocks", "常驻块数", "仅 manual 档案：流式时常驻显存的块数（≥2）。", 4, 2, 64, 1, show={"memory_profile": ["manual"]}),
            flag("act_offload", "Activation Offload", "激活卸载", "仅 manual 档案：把保存的激活值卸载到内存。", True, show={"memory_profile": ["manual"]}),
            sel("optim_state", "Optimizer State", "优化器状态位置", "仅 manual 档案：AdamW 状态放置位置。", "gpu", [
                opt("gpu", "gpu", "显存。"), opt("stream", "stream", "内存，按参数流式传输。"), opt("cpu", "cpu", "内存，在 CPU 上更新。"),
            ], show={"memory_profile": ["manual"]}),
            flag("grad_offload", "Gradient Offload", "梯度卸载", "仅 manual 档案：梯度在主机内存累加。", False, show={"memory_profile": ["manual"]}),
            flag("fused_step", "Fused Step", "融合优化步", "仅 manual 档案：在反向传播中逐参数完成优化器更新。", False, show={"memory_profile": ["manual"]}),
        ]},
        {"title": "Diffusion Objective", "zh": "扩散目标", "tip": "时间步采样、损失函数与加权。", "fields": [
            sel("timestep", "Timestep Sampler", "时间步采样器", "训练时如何抽取噪声时间步。", "moss", [
                opt("moss", "MOSS", "MOSS 官方采样分布（推荐）。"),
                opt("uniform", "Uniform", "均匀采样。"),
                opt("logit_normal", "Logit Normal", "logit-normal 分布采样。"),
            ]),
            num("timestep_mu", "Timestep Mu", "时间步 Mu", "MOSS 时间步分布参数 mu。", -1.09, -10, 10, 0.01, show={"timestep": ["moss"]}),
            num("timestep_s", "Timestep S", "时间步 S", "MOSS 时间步分布参数 s。", 0.95, 0, 10, 0.01, show={"timestep": ["moss"]}),
            num("timestep_shift", "Timestep Shift", "时间步 Shift", "时间步偏移。", 1.0, 0, 20, 0.1, show={"timestep": ["moss"]}),
            num("timestep_uniform_mix", "Uniform Mix", "均匀混合比", "混入均匀分布的比例。", 0.01, 0, 1, 0.01, show={"timestep": ["moss"]}),
            sel("loss", "Loss", "损失函数", "训练损失。", "mse", [
                opt("mse", "MSE", "均方误差（默认）。"), opt("huber", "Huber", "Huber 损失，对离群值更鲁棒。"), opt("l1", "L1", "L1 绝对误差。"),
            ]),
            sel("weighting", "Loss Weighting", "损失加权", "按噪声水平对损失加权。", "none", [
                opt("none", "none", "不加权。"), opt("x0", "x0", "x0 加权。"), opt("eps", "eps", "eps 加权。"),
            ]),
        ]},
        {"title": "Runtime", "zh": "运行时", "tip": "内核调优、编译、日志等。", "fields": [
            sel("tune", "Kernel Tuning", "内核调优", "首次运行时是否测速选择最快内核。", "table", [
                opt("table", "table", "查表，缺失的在训练前自动调优并保存（推荐）。"),
                opt("off", "off", "不调优，使用固定默认内核。"),
            ]),
            sel("compile", "torch.compile", "torch.compile", "是否使用 torch.compile 加速。", "off", [
                opt("off", "off", "关闭。"), opt("block", "block", "按块编译。"), opt("model", "model", "整模型编译（首次很慢）。"),
            ]),
            num("log_every", "Log Every N Steps", "日志间隔步数", "控制台每 N 步输出一行训练状态。", 10, 1, 10000, 1),
            sel("logger", "Logger", "日志记录器", "指标记录方式。", "console", [
                opt("console", "console", "控制台 + metrics.jsonl。"), opt("wandb", "wandb", "同时上传 Weights & Biases。"),
            ]),
            num("nonfinite_patience", "Non-finite Patience", "非有限值容忍步数", "连续出现 NaN/Inf 的步数达到该值就停止训练；0 为关闭。", 8, 0, 1000, 1),
            flag("train_text_refiner", "Train Text Refiner", "训练文本精炼块", "适配器/全量微调也作用于 2 个文本精炼块。"),
            flag("skip_cache_step", "Skip Cache Pass", "跳过缓存步骤", "跳过训练前的离线缓存步骤（已确认缓存完整时可节省启动时间）。"),
        ]},
        {"title": "Custom Overrides", "zh": "自定义覆盖", "tip": "直接追加原生 KohakuMoss 配置项，用于 UI 未提供的参数。", "fields": [
            field("extra_config", "Extra Config (Python)", "额外配置（Python）", "每行一个 KohakuMoss 配置项，写在自动生成的配置最后并覆盖前面的值，例如：\nPLANNER_KWARGS = {'window': 64}\nSAMPLE_SKIP_LAYERS = (22,)\nWindows 路径请自行写成 /mnt/f/... 形式。", "textarea", "", wide=True, placeholder="KNOB = value"),
        ]},
    ]},
]


def all_fields():
    for tab in TABS:
        for section in tab["sections"]:
            yield from section["fields"]


def defaults():
    return {f["key"]: f["default"] for f in all_fields()}
