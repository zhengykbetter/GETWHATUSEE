# video_ocr_module.py
import cv2
import os
import time
import numpy as np
import pynvml
import logging
from typing import Optional, Tuple, List

# ===================== 日志配置（核心改造） =====================
def setup_logger(log_file: str = "video_ocr.log", log_level=logging.INFO):
    """配置日志系统：输出到文件+控制台，分级打印"""
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器（写入日志文件）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器（可选，保留关键信息）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)  # 控制台仅打印警告/错误
    
    # 配置根logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# 初始化日志
logger = setup_logger()

# ===================== 解决模型源检查提示问题 =====================
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
try:
    from paddleocr import PaddleOCR
except ImportError as e:
    logger.error(f"导入PaddleOCR失败：{e}")
    raise

# ===================== 全局配置 =====================
CONFIDENCE_THRESHOLD = 0.8
PURE_TEXT_OUTPUT_FILE = "pure_ocr_text.txt"
SEPARATOR_LINE = "--------------------------------------------------------------------------------"

# ===================== 全局测速统计 =====================
time_stats = {
    "frame_extract": [],
    "ocr_total": [],
    "result_parse": [],
    "frame_save": [],
    "single_frame_total": []
}

# ===================== OCR优化配置接口 =====================
def get_ocr_optimization_config(
    use_lightweight_model: bool = True,
    lightweight_model_type: str = "PP-OCRv5_mobile",
    device: str = "gpu",
    resize_frame: bool = True,
    resize_size: tuple = (960, 640),
    convert_to_gray: bool = True,
    det_thresh: float = 0.6,
    skip_frame_save: bool = False,
    **kwargs
) -> dict:
    """生成OCR优化配置字典（简化参数，保留核心优化项）"""
    optimization_config = {
        "model_optimization": {
            "use_lightweight_model": use_lightweight_model,
            "lightweight_model_type": lightweight_model_type
        },
        "hardware_optimization": {
            "device": device
        },
        "preprocess_optimization": {
            "resize_frame": resize_frame,
            "resize_size": resize_size,
            "convert_to_gray": convert_to_gray
        },
        "runtime_optimization": {
            "det_thresh": det_thresh
        },
        "io_optimization": {
            "skip_frame_save": skip_frame_save
        }
    }
    # 合并自定义参数
    for key, value in kwargs.items():
        for opt_group in optimization_config.values():
            if key in opt_group:
                opt_group[key] = value
                break
    return optimization_config

# ===================== GPU监控工具 =====================
def init_gpu_monitor() -> bool:
    """初始化GPU监控"""
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"GPU监控初始化成功，检测到{gpu_count}块GPU")
        return True
    except Exception as e:
        logger.warning(f"GPU监控初始化失败：{e}")
        return False

def get_gpu_usage(gpu_id: int = 0) -> Tuple[int, float, float]:
    """获取GPU利用率和显存占用"""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / 1024 / 1024
        mem_total = mem_info.total / 1024 / 1024
        return gpu_util, mem_used, mem_total
    except Exception as e:
        logger.error(f"获取GPU信息失败：{e}")
        return -1, -1, -1

def close_gpu_monitor():
    """关闭GPU监控"""
    try:
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.warning(f"关闭GPU监控失败：{e}")

# ===================== 帧预处理优化 =====================
def optimize_frame_preprocessing(frame: np.ndarray, ocr_config: dict) -> np.ndarray:
    """帧预处理：保持长宽比缩放 + 转灰度图"""
    processed_frame = frame.copy()
    preprocess_info = []

    # 1. 保持长宽比缩放
    if ocr_config['preprocess_optimization']['resize_frame']:
        h, w = processed_frame.shape[:2]
        target_w, target_h = ocr_config['preprocess_optimization']['resize_size']
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        processed_frame = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        preprocess_info.append(f"缩放至{new_w}×{new_h}（原{w}×{h}，比例{scale:.2f}）")

    # 2. 转灰度图
    if ocr_config['preprocess_optimization']['convert_to_gray']:
        if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            preprocess_info.append("转灰度图（3通道→1通道）")
        else:
            preprocess_info.append("转灰度图（已为单通道，跳过）")

    logger.info(f"帧预处理完成：{' | '.join(preprocess_info)}")
    return processed_frame

# ===================== OCR模型初始化 =====================
def init_ocr_model(optimization_config: dict) -> Tuple[object, float]:
    """初始化OCR模型"""
    start = time.time()
    det_thresh = optimization_config["runtime_optimization"]["det_thresh"]
    model_type = "mobile版" if optimization_config["model_optimization"]["use_lightweight_model"] else "服务器版"
    logger.info(f"初始化OCR模型（{model_type}），det_thresh={det_thresh}")

    # 基础OCR配置
    ocr_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_det_thresh": det_thresh,
        "device": optimization_config["hardware_optimization"]["device"]
    }

    # 启用mobile模型
    if optimization_config["model_optimization"]["use_lightweight_model"]:
        model_type = optimization_config["model_optimization"]["lightweight_model_type"]
        ocr_kwargs["text_detection_model_name"] = f"{model_type}_det"
        ocr_kwargs["text_recognition_model_name"] = f"{model_type}_rec"
    else:
        ocr_kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
        ocr_kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"

    ocr = PaddleOCR(**ocr_kwargs)
    model_load_time = time.time() - start
    logger.info(f"OCR模型加载完成，耗时{model_load_time:.2f}秒")
    return ocr, model_load_time

# ===================== OCR结果解析 =====================
def safe_parse_ocr_result(result, frame_filename: str) -> Tuple[List[str], List[str]]:
    """安全解析OCR结果"""
    parsed_lines = []
    pure_text_lines = []
    logger.debug(f"开始解析{frame_filename}的OCR结果，原始类型：{type(result)}")

    try:
        if not result or len(result) == 0:
            logger.warning(f"{frame_filename}：OCR返回空结果")
            return ["[OCR返回空结果]"], ["[OCR返回空结果]"]

        ocr_result = result[0] if isinstance(result, list) else result

        # 兼容新版OCRResult对象和旧版格式
        if hasattr(ocr_result, 'rec_texts') and hasattr(ocr_result, 'rec_scores'):
            rec_texts = ocr_result.rec_texts
            rec_scores = ocr_result.rec_scores
        elif isinstance(ocr_result, dict) and 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
            rec_texts = ocr_result['rec_texts']
            rec_scores = ocr_result['rec_scores']
        else:
            logger.debug(f"{frame_filename}：使用旧版列表解析逻辑")
            frame_result = result[0] if isinstance(result, list) else result
            if not frame_result or len(frame_result) == 0:
                return ["[当前帧无有效文字]"], ["[当前帧无有效文字]"]
            for line in frame_result:
                if not isinstance(line, list) or len(line) < 2:
                    continue
                text_part = line[1]
                if isinstance(text_part, (list, tuple)) and len(text_part) >= 2:
                    text = text_part[0].strip() if text_part[0] else ""
                    score = float(text_part[1]) if text_part[1] else 0.0
                    if text:
                        if score >= CONFIDENCE_THRESHOLD:
                            parsed_lines.append(f"[{score:.2f}] {text}")
                            pure_text_lines.append(text)
                        else:
                            logger.debug(f"{frame_filename}：文字置信度过低（{score:.4f} < {CONFIDENCE_THRESHOLD}），过滤：{text}")
            return parsed_lines or ["[当前帧无有效文字]"], pure_text_lines or ["[当前帧无有效文字]"]

        # 验证结果格式
        if not isinstance(rec_texts, list) or not isinstance(rec_scores, list):
            logger.error(f"{frame_filename}：rec_texts/rec_scores非列表格式")
            return ["[OCR结果格式异常]"], ["[OCR结果格式异常]"]
        if len(rec_texts) != len(rec_scores):
            logger.error(f"{frame_filename}：文字和置信度数量不匹配")
            return ["[OCR结果格式异常]"], ["[OCR结果格式异常]"]

        logger.debug(f"{frame_filename}：识别到{len(rec_texts)}个文字片段")
        # 过滤低置信度文字
        for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
            text = str(text).strip() if text else ""
            score = float(score) if score else 0.0
            if text and score >= CONFIDENCE_THRESHOLD:
                parsed_lines.append(f"[{score:.2f}] {text}")
                pure_text_lines.append(text)
            elif text:
                logger.debug(f"{frame_filename}：片段{idx}置信度过低，过滤：{text}（{score:.4f}）")
            else:
                logger.debug(f"{frame_filename}：片段{idx}为空文字，过滤")

        if not parsed_lines:
            parsed_lines = ["[当前帧无有效文字]"]
            pure_text_lines = ["[当前帧无有效文字]"]
            logger.warning(f"{frame_filename}：无有效文字")

        logger.debug(f"{frame_filename}解析完成：带置信度={parsed_lines} | 纯净版={pure_text_lines}")
        return parsed_lines, pure_text_lines

    except Exception as e:
        error_msg = f"[解析异常] {str(e)[:50]}"
        logger.error(f"{frame_filename}解析失败：{e}", exc_info=True)
        return [error_msg], [error_msg]

# ===================== 核心OCR处理函数（对外暴露） =====================
def run_video_ocr(
    video_path: str,
    output_dir: str = "video_ocr_results",
    optimization_config: Optional[dict] = None
) -> str:
    """
    执行视频OCR识别（模块核心函数）
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param optimization_config: 优化配置字典（None则使用默认配置）
    :return: 纯净文本输出文件的路径
    """
    # 初始化配置
    if optimization_config is None:
        optimization_config = get_ocr_optimization_config()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    frame_save_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_save_dir, exist_ok=True)
    result_txt = os.path.join(output_dir, "ocr_results.txt")
    speed_analysis_txt = os.path.join(output_dir, "speed_analysis.txt")
    pure_text_txt = os.path.join(output_dir, PURE_TEXT_OUTPUT_FILE)

    # 打开视频
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件：{video_path}")
        raise ValueError(f"无法打开视频文件：{video_path}")

    # 初始化GPU监控
    gpu_monitor_enabled = init_gpu_monitor()

    # 视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info(f"视频信息：帧率={fps:.1f} | 总帧数={total_frames} | 总时长={duration:.1f}秒")
    logger.info(f"输出目录：{output_dir} | 置信度阈值={CONFIDENCE_THRESHOLD}")

    # 初始化模型
    ocr, model_load_time = init_ocr_model(optimization_config)

    # 初始化结果文件
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"视频路径：{video_path}\n")
        f.write(f"优化配置：{optimization_config}\n")
        f.write("="*80 + "\n\n")

    # 初始化纯净文本文件
    with open(pure_text_txt, "w", encoding="utf-8") as f:
        f.write(f"视频OCR纯净版文字结果（置信度≥{CONFIDENCE_THRESHOLD}）\n")
        f.write(f"{SEPARATOR_LINE}\n")

    frame_count = 0
    saved_frame_num = 0

    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps
        # 每秒提取1帧
        if int(current_time) == saved_frame_num and frame_count % int(fps) == 0:
            frame_extract_start = time.time()
            frame_filename = f"frame_{int(current_time)}s_{saved_frame_num}.jpg"
            frame_path = os.path.join(frame_save_dir, frame_filename)

            logger.info(f"处理第{saved_frame_num+1}帧：{frame_filename}，原始分辨率={frame.shape[:2]}")

            # 帧预处理
            frame = optimize_frame_preprocessing(frame, optimization_config)

            # 保存帧
            frame_save_start = time.time()
            save_success = True
            if not optimization_config["io_optimization"]["skip_frame_save"]:
                save_success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_save_time = time.time() - frame_save_start
            time_stats["frame_save"].append(frame_save_time)
            if save_success:
                logger.debug(f"帧保存完成：{frame_path}，耗时{frame_save_time:.3f}s")
            else:
                logger.warning(f"帧保存失败：{frame_path}")

            frame_extract_time = time.time() - frame_extract_start
            time_stats["frame_extract"].append(frame_extract_time)

            # OCR识别
            ocr_start = time.time()
            infer_frame = frame_path if not optimization_config["io_optimization"]["skip_frame_save"] else frame
            raw_result = ocr.ocr(infer_frame)
            ocr_total_time = time.time() - ocr_start
            time_stats["ocr_total"].append(ocr_total_time)
            logger.info(f"{frame_filename} OCR识别完成，耗时{ocr_total_time:.3f}s")

            # GPU状态监控
            if gpu_monitor_enabled:
                gpu_util, mem_used, mem_total = get_gpu_usage()
                if gpu_util != -1:
                    logger.info(f"GPU状态：利用率={gpu_util}% | 显存={mem_used:.0f}/{mem_total:.0f}MB")

            # 解析结果
            parse_start = time.time()
            full_text_lines, pure_text_lines = safe_parse_ocr_result(raw_result, frame_filename)
            parse_time = time.time() - parse_start
            time_stats["result_parse"].append(parse_time)
            logger.info(f"{frame_filename} 结果解析完成，耗时{parse_time:.3f}s")

            # 总耗时
            single_frame_total_time = frame_extract_time + ocr_total_time + parse_time
            time_stats["single_frame_total"].append(single_frame_total_time)

            # 保存结果
            with open(result_txt, "a", encoding="utf-8") as f:
                f.write(f"📌 帧：{frame_filename} | 时间：{current_time:.1f}秒\n")
                f.write(f"⏱️  耗时：帧提取={frame_extract_time:.3f}s | OCR={ocr_total_time:.3f}s | 解析={parse_time:.3f}s | 总={single_frame_total_time:.3f}s\n")
                f.write(f"📝 识别文字（带置信度）：{' | '.join(full_text_lines)}\n")
                f.write(f"📝 识别文字（纯净版）：{' | '.join(pure_text_lines)}\n")
                f.write("-"*80 + "\n\n")

            # 仅保留纯净文本的文件输出（核心输出）
            with open(pure_text_txt, "a", encoding="utf-8") as f:
                if pure_text_lines and pure_text_lines[0] != "[当前帧无有效文字]":
                    f.write(f"{' | '.join(pure_text_lines)}\n")
                else:
                    f.write("[当前帧无有效文字]\n")
                f.write(f"{SEPARATOR_LINE}\n")

            saved_frame_num += 1

        frame_count += 1

    # 生成性能报告
    with open(speed_analysis_txt, "w", encoding="utf-8") as f:
        f.write("📊 PP-OCRv5 视频识别性能分析报告\n")
        f.write("="*80 + "\n")
        f.write(f"基础信息：\n")
        f.write(f"  - 视频：{video_path} | 帧率={fps:.1f} | 提取帧数={saved_frame_num}\n")
        f.write(f"  - 模型加载耗时：{model_load_time:.2f}秒\n")
        f.write(f"  - 置信度阈值={CONFIDENCE_THRESHOLD}\n")
        f.write(f"\n各环节平均耗时（秒/帧）：\n")
        for key in time_stats:
            if time_stats[key]:
                avg = np.mean(time_stats[key])
                max_v = np.max(time_stats[key])
                min_v = np.min(time_stats[key])
                f.write(f"  - {key.replace('_', ' ')}：平均={avg:.3f} | 最大={max_v:.3f} | 最小={min_v:.3f}\n")

        # GPU统计
        if gpu_monitor_enabled:
            gpu_util, mem_used, mem_total = get_gpu_usage()
            if gpu_util != -1:
                f.write(f"\n📊 GPU资源统计：\n")
                f.write(f"  - GPU利用率：{gpu_util}% | 显存占用：{mem_used:.0f}/{mem_total:.0f}MB\n")
                if gpu_util < 50:
                    f.write(f"  - 提示：GPU利用率低于50%，可启用批量推理放大GPU优势\n")

        # 瓶颈分析
        avg_times = {k: np.mean(v) if v else 0 for k, v in time_stats.items()}
        bottleneck = max(avg_times, key=avg_times.get) if avg_times else "none"
        f.write(f"\n⚠️  性能瓶颈：{bottleneck.replace('_', ' ')}（平均 {avg_times[bottleneck]:.3f}秒/帧）\n")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    if gpu_monitor_enabled:
        close_gpu_monitor()

    logger.info(f"处理完成！共提取{saved_frame_num}帧，纯净文本文件：{pure_text_txt}")
    return pure_text_txt

# 模块入口保护
if __name__ == "__main__":
    # 模块自测（可选）
    test_video_path = "./test_video.mp4"
    try:
        pure_text_path = run_video_ocr(test_video_path)
        print(f"自测完成，纯净文本文件：{pure_text_path}")
    except Exception as e:
        logger.error(f"模块自测失败：{e}", exc_info=True)