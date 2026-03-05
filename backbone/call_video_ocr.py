# call_video_ocr.py
import sys
import os
import locale
import unicodedata
import requests
import json
import configparser  # 核心修复：导入configparser
from typing import List, Dict

# ===================== 全局配置（可直接修改） =====================
USE_ITERATIVE_SUMMARY = True
CHUNK_SIZE = 2000
SUB_SUMMARY_MAX_LEN = 100
FINAL_SUMMARY_MAX_LEN = 200
LLM_CONFIG = {
    "api_key": "",
    "base_url": "",
    "model_name": "deepseek-chat"
}

# ===================== 第一步：强制系统/Python全链路UTF-8 =====================
def setup_encoding():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'zh_CN.UTF-8'
    os.environ['LC_ALL'] = 'zh_CN.UTF-8'
    os.environ['LC_CTYPE'] = 'zh_CN.UTF-8'
    os.environ['LC_MESSAGES'] = 'zh_CN.UTF-8'
    print("="*80)
    print("📌 编码环境信息：")
    print(f"Python默认编码: {sys.getdefaultencoding()}")
    print(f"系统首选编码: {locale.getpreferredencoding()}")
    print("="*80)

# ===================== 第二步：OCR处理器 =====================
class OCRProcessor:
    def __init__(self, video_path: str, output_dir: str = "video_ocr_results"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.video_filename = os.path.basename(video_path).replace(".mp4", "")
        self.ocr_file_path = os.path.join(output_dir, f"{self.video_filename}_ocr.txt")
        os.makedirs(output_dir, exist_ok=True)

    def clean_old_ocr_file(self):
        if os.path.exists(self.ocr_file_path):
            os.remove(self.ocr_file_path)
            print(f"🗑️ 已删除旧OCR文件：{self.ocr_file_path}")

    def run_ocr(self, optim_config: Dict):
        from video_ocr_module import run_video_ocr
        temp_ocr_path = run_video_ocr(
            video_path=self.video_path,
            output_dir=self.output_dir,
            optimization_config=optim_config
        )
        os.rename(temp_ocr_path, self.ocr_file_path)
        print(f"✅ OCR处理完成，新OCR文件：{self.ocr_file_path}")
        return self.ocr_file_path

    def read_ocr_text(self) -> str:
        if not os.path.exists(self.ocr_file_path):
            raise FileNotFoundError(f"OCR文件不存在：{self.ocr_file_path}")
        with open(self.ocr_file_path, "r", encoding="utf-8") as f:
            ocr_text = f.read().strip()
        print(f"\n📄 本次OCR文本长度：{len(ocr_text)} 字符")
        print(f"📄 OCR文本前200字符：{ocr_text[:200]}...")
        return ocr_text

# ===================== 第三步：LLM客户端 =====================
class LLMClient:
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.llm_config = self.load_config()
        self._validate_config()

    def load_config(self) -> Dict:
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path, encoding="utf-8")
            llm_config = {
                "api_key": config.get("LLM", "api_key", fallback="").strip(),
                "base_url": config.get("LLM", "base_url", fallback="").strip(),
                "model_name": config.get("LLM", "model_name", fallback="").strip()
            }
        else:
            llm_config = LLM_CONFIG.copy()
        for key in llm_config:
            if not llm_config[key] and LLM_CONFIG[key]:
                llm_config[key] = LLM_CONFIG[key]
        return llm_config

    def _validate_config(self):
        if not self.llm_config["api_key"]:
            raise ValueError("LLM配置错误：api_key不能为空（请检查config.ini或代码LLM_CONFIG）")
        if not self.llm_config["base_url"].startswith(("http://", "https://")):
            raise ValueError(f"LLM配置错误：base_url格式错误（当前：{self.llm_config['base_url']}）")
        print(f"✅ LLM配置加载成功：model={self.llm_config['model_name']}")

    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        clean_chars = []
        for char in text:
            if char.isprintable() or char in '\n\r\t':
                clean_chars.append(char)
            else:
                clean_chars.append(' ')
        clean_text = ''.join(clean_chars).strip()
        return clean_text.encode('utf-8', errors='replace').decode('utf-8')

    def send_request(self, prompt: str) -> str:
        prompt = self.clean_text(prompt)
        api_url = f"{self.llm_config['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config['api_key']}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8"
        }
        request_body = {
            "model": self.llm_config["model_name"],
            "messages": [
                {"role": "system", "content": "你是专业的视频OCR文本总结助手，仅基于本次提供的文本总结，忽略所有过往内容。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": False,
            "n": 1
        }
        try:
            request_data = json.dumps(request_body, ensure_ascii=False).encode('utf-8')
            response = requests.post(
                api_url, headers=headers, data=request_data, timeout=30, verify=True
            )
            response.encoding = 'utf-8'
            if response.status_code != 200:
                raise RuntimeError(f"LLM请求失败：状态码{response.status_code} | 响应：{response.text[:200]}")
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            raise RuntimeError(f"LLM调用失败：{str(e)} | Prompt片段：{prompt[:50]}")

# ===================== 第四步：迭代式精炼总结器 =====================
class IterativeSummarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.chunk_size = CHUNK_SIZE
        self.sub_summary_max_len = SUB_SUMMARY_MAX_LEN
        self.final_summary_max_len = FINAL_SUMMARY_MAX_LEN

    def split_text_to_chunks(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i+self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        print(f"\n🔪 长文本拆分完成，共生成 {len(chunks)} 个文本块")
        return chunks

    def generate_sub_summary(self, chunk: str, chunk_index: int) -> str:
        prompt = f"""
请总结以下视频OCR文本块的核心内容，严格遵守：
1. 仅总结本次提供的文本块，忽略所有其他信息；
2. 保留该块的完整上下文，不丢失关键信息；
3. 总结字数控制在{self.sub_summary_max_len}字以内；
4. 标注该块的序号（第{chunk_index+1}块），语言简洁流畅。

文本块内容：
{chunk}
        """.strip()
        sub_summary = self.llm_client.send_request(prompt)
        print(f"📝 第{chunk_index+1}块子总结：{sub_summary}")
        return sub_summary

    def aggregate_final_summary(self, sub_summaries: List[str]) -> str:
        sub_summaries_text = "\n".join([f"第{i+1}块：{summary}" for i, summary in enumerate(sub_summaries)])
        prompt = f"""
请基于以下所有子总结，聚合生成视频OCR文本的最终总结，严格遵守：
1. 整合所有子总结的核心信息，保留完整上下文，不丢失关键内容；
2. 按文本块的顺序（时间轴）组织总结逻辑；
3. 总结字数控制在{self.final_summary_max_len}字以内；
4. 仅基于本次子总结，不添加任何编造信息，忽略所有过往内容。

所有子总结：
{sub_summaries_text}
        """.strip()
        final_summary = self.llm_client.send_request(prompt)
        return final_summary

    def run_iterative_summary(self, ocr_text: str) -> str:
        chunks = self.split_text_to_chunks(ocr_text)
        if len(chunks) == 0:
            raise ValueError("OCR文本为空，无法生成总结")
        sub_summaries = []
        for i, chunk in enumerate(chunks):
            sub_summary = self.generate_sub_summary(chunk, i)
            sub_summaries.append(sub_summary)
        final_summary = self.aggregate_final_summary(sub_summaries)
        print(f"\n🏁 迭代式总结完成，最终总结：\n{final_summary}")
        return final_summary

# ===================== 第五步：主流程控制器 =====================
class MainWorkflow:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.ocr_processor = OCRProcessor(video_path)
        self.llm_client = LLMClient()
        self.iterative_summarizer = IterativeSummarizer(self.llm_client)
        self.summary_file_path = os.path.join(
            self.ocr_processor.output_dir,
            f"{self.ocr_processor.video_filename}_summary.txt"
        )

    def get_ocr_optim_config(self) -> Dict:
        from video_ocr_module import get_ocr_optimization_config
        return get_ocr_optimization_config(
            resize_size=(1280, 720),
            skip_frame_save=False,
            det_thresh=0.7
        )

    def run_single_summary(self, ocr_text: str) -> str:
        prompt = f"""
请总结以下视频OCR文本内容，严格遵守：
1. 保留完整上下文，核心信息无遗漏；
2. 总结字数控制在{FINAL_SUMMARY_MAX_LEN}字以内；
3. 仅基于本次文本，忽略所有过往内容。

OCR文本：
{ocr_text}
        """.strip()
        summary = self.llm_client.send_request(prompt)
        return summary

    def save_summary(self, summary: str):
        with open(self.summary_file_path, "w", encoding="utf-8") as f:
            f.write(f"📝 视频 {self.ocr_processor.video_filename}.mp4 OCR文本总结\n")
            f.write("="*80 + "\n\n")
            f.write(summary)
        print(f"\n💾 总结已保存至：{self.summary_file_path}")

    def run(self):
        try:
            self.ocr_processor.clean_old_ocr_file()
            optim_config = self.get_ocr_optim_config()
            self.ocr_processor.run_ocr(optim_config)
            ocr_text = self.ocr_processor.read_ocr_text()
            if USE_ITERATIVE_SUMMARY:
                print("\n🚀 启用迭代式精炼总结（解决长文本上下文丢失）")
                summary = self.iterative_summarizer.run_iterative_summary(ocr_text)
            else:
                print("\n🚀 启用单次总结（原有逻辑）")
                summary = self.run_single_summary(ocr_text)
            self.save_summary(summary)
            print("\n" + "="*80)
            print("🎉 程序执行完成！")
            print(f"📄 视频文件：{self.video_path}")
            print(f"📝 总结内容：\n{summary}")
            print("="*80)
        except Exception as e:
            print(f"\n❌ 程序执行失败：{str(e)}")
            sys.exit(1)

# ===================== 程序入口（核心修复：简化警告屏蔽） =====================
if __name__ == "__main__":
    # 修复：直接忽略所有requests相关警告（简单有效，避免属性错误）
    import warnings
    warnings.filterwarnings("ignore")  # 全局忽略警告（或仅忽略DeprecationWarning/RequestsDependencyWarning）
    
    # 1. 设置编码
    setup_encoding()
    # 2. 配置视频路径
    VIDEO_PATH = "./test_video2.mp4"
    # 3. 执行主流程
    workflow = MainWorkflow(VIDEO_PATH)
    workflow.run()