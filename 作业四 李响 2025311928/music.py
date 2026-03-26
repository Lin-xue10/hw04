"""
开源语音识别本地实现
方案：OpenAI Whisper
功能：1. 本地音频文件识别  2. 麦克风实时语音识别
"""
import whisper
import pyaudio
import wave
import os
import time

# 模型配置（笔记本推荐 small，平衡速度与精度）
MODEL_TYPE = "small"
# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
TEMP_FILE = "temp.wav"

def load_whisper_model():
    """加载Whisper模型"""
    print(f"[信息] 正在加载模型：{MODEL_TYPE}")
    model = whisper.load_model(MODEL_TYPE)
    print("[信息] 模型加载完成！")
    return model

def recognize_file(model, audio_path):
    """识别本地音频文件"""
    if not os.path.exists(audio_path):
        print("[错误] 文件不存在！")
        return

    print(f"\n[文件识别] 正在识别：{audio_path}")
    start_time = time.time()
    result = model.transcribe(audio_path, language="zh")
    cost_time = round(time.time() - start_time, 2)

    print(f"识别结果：{result['text']}")
    print(f"识别耗时：{cost_time} 秒")
    return result["text"]

def microphone_recognize(model):
    """麦克风录音并识别"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print("\n[实时识别] 请说话，录音5秒...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("[信息] 录音结束，开始识别...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存临时音频
    wf = wave.open(TEMP_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return recognize_file(model, TEMP_FILE)

if __name__ == "__main__":
    # 加载模型
    asr_model = load_whisper_model()

    while True:
        print("\n===== Whisper 语音识别工具 =====")
        print("1 → 识别本地音频文件")
        print("2 → 麦克风实时语音识别")
        print("0 → 退出程序")
        choice = input("请输入功能序号：")

        if choice == "1":
            path = input("请输入音频文件路径：")
            recognize_file(asr_model, path)
        elif choice == "2":
            microphone_recognize(asr_model)
        elif choice == "0":
            print("[信息] 程序已退出")
            break
        else:
            print("[错误] 输入无效，请重新选择")