import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def generate_wav_from_txt(txt_path, wav_path, sample_rate=44100):
    # 1. テキストファイルから振幅データを読み込む
    print(f"Loading data from {txt_path}...")
    with open(txt_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    
    # NumPy配列に変換
    data = np.array(data, dtype=np.float32)
    
    # 2. データを正規化（最大値を-1.0〜1.0の範囲に収める）
    max_val = np.max(np.abs(data))
    if max_val > 0:
        normalized_data = data / max_val
    else:
        normalized_data = data
        
    # 3. 16ビットPCMフォーマット（-32768〜32767）に変換
    audio_data = (normalized_data * 32767).astype(np.int16)
    
    # 4. WAVファイルとして保存
    print(f"Saving audio to {wav_path} (Sample Rate: {sample_rate}Hz)...")
    wav.write(wav_path, sample_rate, audio_data)
    print("Audio save done!")
    
    # 5. スペクトログラムの描画（周波数分解能を上げて描画）
    plot_spectrogram(audio_data, sample_rate)

def plot_spectrogram(audio_data, sample_rate):
    print("Generating high-resolution spectrogram...")
    
    plt.figure(figsize=(10, 6))
    
    # 周波数分解能を上げるため、NFFTを大きい値（2048）に変更
    # 時間解像度の大幅な低下を防ぐため、noverlapも大きめ（1024や1536など）に調整します
    nfft_val = 2048
    noverlap_val = 1536  # 75% オーバーラップ
    
    # スペクトログラムの計算と描画
    plt.specgram(audio_data, NFFT=nfft_val, Fs=sample_rate, noverlap=noverlap_val, cmap='viridis')
    
    # タイトルとラベルの設定
    plt.title("Spectrogram (High Resolution: 500Hz - 5kHz)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Frequency [Hz]")
    
    # ★ 周波数軸の表示範囲を 0Hz 〜 10000Hz (10kHz) に制限
    plt.ylim(0, 10000)

    # カラーバーを表示
    plt.colorbar(label="Intensity [dB]")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    txt_file = "GenSound1.txt"
    wav_file = "string_vibration.wav"
    
    # 一般的なサンプリングレート（44100Hz）で生成
    generate_wav_from_txt(txt_file, wav_file, sample_rate=44100)