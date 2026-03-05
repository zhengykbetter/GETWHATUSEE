# call_video_ocr.py
import sys
import os
import locale
import unicodedata
import requests
import json

# ===================== 第一步：强制系统/Python全链路UTF-8 =====================
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'zh_CN.UTF-8'
os.environ['LC_ALL'] = 'zh_CN.UTF-8'
os.environ['LC_CTYPE'] = 'zh_CN.UTF-8'
os.environ['LC_MESSAGES'] = 'zh_CN.UTF-8'

# 验证编码配置
print("="*80)
print("📌 编码环境调试信息：")
print(f"Python默认编码: {sys.getdefaultencoding()}")
print(f"系统首选编码: {locale.getpreferredencoding()}")
print("="*80)

# ===================== 第二步：导入依赖 =====================
sys.path.append("/usr/lib/python3/dist-packages")

# 验证distro导入（非核心）
try:
    import distro
    print(f"✅ 成功导入distro，版本：{distro.__version__}")
except ImportError:
    print("⚠️ 未导入distro（不影响核心功能），执行：pip install --user distro")

import configparser
from video_ocr_module import run_video_ocr, get_ocr_optimization_config

# ===================== 工具函数：文本清理 =====================
def clean_text_for_encoding(text: str) -> str:
    """清理文本，过滤特殊字符，确保UTF-8兼容"""
    text = unicodedata.normalize('NFKC', text)
    clean_chars = []
    for char in text:
        if (
            '\u4e00' <= char <= '\u9fff' or  # 中文
            char.isalnum() or                # 字母/数字
            char in '，。！？：；""''()（）[]【】{}、·@#￥%&*+-=<>《》 \n' or  # 常用标点
            char.isspace()
        ):
            clean_chars.append(char)
        else:
            clean_chars.append(' ')
    clean_text = ''.join(clean_chars).strip()
    return clean_text.encode('utf-8', errors='replace').decode('utf-8')

# ===================== LLM配置加载（核心修复：清理URL空格） =====================
def load_llm_config(config_path: str = "config.ini") -> dict:
    config = configparser.ConfigParser()
    if not config.read(config_path, encoding="utf-8"):
        raise FileNotFoundError(f"无法读取配置文件：{config_path}")
    
    if "LLM" not in config.sections():
        raise KeyError(f"配置文件缺少 [LLM] 节")
    
    # 核心修复：清理所有配置项的首尾空格（尤其是base_url）
    llm_config = {
        "api_key": config.get("LLM", "api_key", fallback="").strip(),  # 清理API密钥空格
        "base_url": config.get("LLM", "base_url", fallback="").strip(),  # 清理URL空格
        "model_name": config.get("LLM", "model_name", fallback="").strip()  # 清理模型名空格
    }
    
    # 验证配置
    if not llm_config["api_key"]:
        raise ValueError("LLM配置中api_key不能为空（请检查config.ini）")
    if not llm_config["base_url"]:
        raise ValueError("LLM配置中base_url不能为空（如https://api.deepseek.com/v1）")
    if not llm_config["model_name"]:
        raise ValueError("LLM配置中model_name不能为空（如deepseek-chat）")
    
    # 验证base_url格式（避免缺少https://）
    if not llm_config["base_url"].startswith(("http://", "https://")):
        raise ValueError(f"base_url格式错误：必须以http://或https://开头（当前值：{llm_config['base_url']}）")
    
    print(f"✅ LLM配置加载成功：base_url={llm_config['base_url']}，model_name={llm_config['model_name']}")
    return llm_config

# ===================== 核心总结函数（修复URL拼接） =====================
def summarize_pure_text(pure_text_path: str, config_path: str = "config.ini") -> str:
    llm_config = load_llm_config(config_path)
    
    # ===== 新增：验证读取的文件和内容 =====
    print(f"\n🔍 正在读取的OCR文本文件：{pure_text_path}")
    print(f"🔍 文件是否存在：{os.path.exists(pure_text_path)}")
    # 输出文件前300字符，确认是新视频的内容
    with open(pure_text_path, "r", encoding="utf-8") as f:
        first_300 = f.read()[:300]
    print(f"🔍 OCR文本前300字符：{first_300}")
    # ======================================
    
    # 1. 读取并清理OCR文本
    if not os.path.exists(pure_text_path):
        raise FileNotFoundError(f"OCR文本文件不存在：{pure_text_path}")
    
    with open(pure_text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    clean_text = clean_text_for_encoding(raw_text)
    print(f"\n📌 清理后OCR文本长度：{len(clean_text)} 字符")
    print(f"📌 OCR文本片段：{clean_text[:100]}...")
    
    # 2. 构建Prompt
    prompt = f"""
请严格按照以下要求总结视频OCR识别的文本内容：
0. 仅总结本次提供的 OCR 文本，完全忽略之前的所有内容
1. 仅总结文本中**实际存在的内容**，不要添加任何编造的信息，不要遗漏核心信息；
2. 过滤无意义内容：如"[当前帧无有效文字]"、重复的分隔线、空白字符等；
3. 总结字数控制在200字以内，语言简洁、流畅，符合中文表达习惯；
4. 若文本是对话/字幕，提炼核心对话主题和关键人物；若文本是零散内容，归纳主要事件/信息；
5. 保留文本中的关键数字、名称、时间等核心信息，不要过滤。

OCR识别的原始文本内容：
{clean_text}
    """.strip()
    
    # 3. 手动构造LLM请求（修复URL拼接）
    # 核心：正确拼接API地址，避免多余字符
    api_url = f"{llm_config['base_url'].rstrip('/')}/chat/completions"
    print(f"\n📌 调用LLM API地址：{api_url}")  # 调试：输出最终API地址
    
    # 请求头：强制UTF-8
    headers = {
        "Authorization": f"Bearer {llm_config['api_key']}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8"
    }
    
    # 请求体
    request_body = {
        "model": llm_config["model_name"],
        "messages": [
            {"role": "system", "content": "你是专业的视频OCR文本总结助手，总结结果简洁、准确、易懂。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        # 强制UTF-8编码请求体
        request_data = json.dumps(request_body, ensure_ascii=False).encode('utf-8')
        
        # 发送请求（增加超时和重试，提升稳定性）
        response = requests.post(
            api_url,
            headers=headers,
            data=request_data,
            timeout=30,
            verify=True  # 验证SSL证书（避免证书错误）
        )
        
        # 解析响应
        response.encoding = 'utf-8'
        if response.status_code != 200:
            raise RuntimeError(f"LLM API请求失败：状态码{response.status_code} | 响应：{response.text[:200]}")
        
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        return clean_text_for_encoding(summary)
    
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"API连接失败：请检查网络或base_url是否正确（当前：{api_url}） | 错误：{e}")
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"API请求超时：请检查网络或增加超时时间 | 错误：{e}")
    except Exception as e:
        raise RuntimeError(f"调用大模型失败：{str(e)} | Prompt片段：{prompt[:50]}")

# ===================== 主流程 =====================
# 在main函数中，修改OCR文本路径的生成逻辑（关键！）
def main():
    # 配置参数
    VIDEO_PATH = "./test_video.mp4"       
    OUTPUT_DIR = "video_ocr_results"     
    CONFIG_PATH = "config.ini"           
    
    # ===== 核心修改：按视频名生成唯一的OCR输出文件 =====
    # 提取视频文件名（不带路径、不带后缀）
    video_filename = os.path.basename(VIDEO_PATH).replace(".mp4", "")
    # 生成唯一的OCR文本路径（比如 test_video2_ocr.txt）
    pure_text_path = os.path.join(OUTPUT_DIR, f"{video_filename}_ocr.txt")
    # 生成唯一的总结文件
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, f"{video_filename}_summary.txt")
    # ==================================================
    
    # OCR配置
    optim_config = get_ocr_optimization_config(
        resize_size=(1280, 720),
        skip_frame_save=False,
        det_thresh=0.7
    )
    
    try:
        print("="*80)
        print(f"开始处理视频：{VIDEO_PATH}")
        print(f"📌 本次OCR文本将保存至：{pure_text_path}")  # 输出文件路径，确认唯一性
        print("="*80)
        
        # ===== 强制重新生成OCR文本（覆盖旧文件） =====
        # 先删除旧的OCR文件（如果存在），确保重新生成
        if os.path.exists(pure_text_path):
            os.remove(pure_text_path)
            print(f"🗑️ 已删除旧的OCR文件：{pure_text_path}")
        
        # 调用OCR模块，指定输出到唯一路径
        # 注意：如果run_video_ocr函数需要传入输出路径，修改调用方式：
        # pure_text_path = run_video_ocr(
        #     video_path=VIDEO_PATH,
        #     output_dir=OUTPUT_DIR,
        #     output_filename=f"{video_filename}_ocr.txt",  # 传给OCR函数
        #     optimization_config=optim_config
        # )
        # 如果run_video_ocr函数不支持指定输出文件名，手动替换：
        # 先调用OCR（生成默认文件），再移动/重命名为唯一文件
        temp_ocr_path = run_video_ocr(
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            optimization_config=optim_config
        )
        # 移动并覆盖为唯一文件
        os.rename(temp_ocr_path, pure_text_path)
        temp_ocr_path = pure_text_path  # 同步路径
        # ==============================================
        
        print(f"\n✅ OCR处理完成！本次OCR文本路径：{pure_text_path}")
        
        # 后续调用LLM时，传入这个唯一的pure_text_path
        print("\n" + "="*80)
        print("开始调用大模型总结OCR文本...")
        print("="*80)
        summary_text = summarize_pure_text(pure_text_path, CONFIG_PATH)
        
        # 保存总结到唯一文件
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(f"📝 视频 {video_filename}.mp4 OCR文本总结\n")
            f.write("="*80 + "\n\n")
            f.write(summary_text)
        
        # 输出结果
        print(f"\n✅ 总结成功！本次总结文件：{SUMMARY_FILE}")
        print(f"\n📝 核心总结：")
        print("-"*80)
        print(summary_text)
        print("-"*80)
        
    except Exception as e:
        print(f"\n❌ 执行失败：{e}")
        sys.exit(1)

# ===================== 兼容OCR模块返回值 =====================
if __name__ == "__main__":
    import types
    def patch_run_video_ocr(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result[0] if isinstance(result, (tuple, list)) else result
        return wrapper
    run_video_ocr = patch_run_video_ocr(run_video_ocr)
    main()