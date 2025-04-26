import os
import time
import numpy as np
import pandas as pd
from kyber import Kyber512  #  Kyber768 & Kyber1024
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.stats import pearsonr

# CSV file path
csv_path = "MitEcgTypeAnls/v_type_concatenated_107.csv"  #  ECG CSV path

# Encrypted block size (must be the length of bytes that Kyber512 can handle)
chunk_size = 8

# encrypted file path
encrypted_file = "ecg_encrypted_type.npy"
decrypted_file = "ecg_decrypted_type.npy"

# read csv path
print("Loading ECG data in ...")
df = pd.read_csv(csv_path)

# turn ECG signal to NumPy arrays (float32)
ecg_data = df.values.astype(np.float32)  # df此时是pandas读取的DataFrame表格，而.drop是pandas中的典型用法。还有.values 是把 DataFrame 转成 NumPy 数组。.astype(np.float32) 是把数据类型转成 float32。
# df.drop(labels, axis=0 or 1, inplace=False) 默认axis=0删除行，axis=0删除列，也可设置columns明确删除列，例如:.drop(columns=['labels'])
# labels = df['labels'].values  # labels

# # Display of selected data
# print("Example of ECG data:")
# print(df.head())

# Combining signal data into one-dimensional arrays (for ease of encryption)
ecg_flat = ecg_data.flatten()
#此时ecg_data就是NumPy的数组.flatten()用于将多维数组展平为 一维数组
#numpy数组还可用.shape和Number of dimensions(.ndim)来看有几个维度


# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.title("Original ECG")
# plt.plot(ecg_flat, label="Original ECG", color='blue')
# plt.legend()
# plt.show()
#
# ========== Kyber512 keygen ==========
print("\nGenerate Kyber512 key pair...")
public_key, secret_key = Kyber512.keygen()

# ========== Encrypted ECG Data ==========
print("\nStarted encrypting ECG signals...")

ciphertexts = []
encrypted_chunks = []

# Record encryption time
encryption_start_time = time.time()

for i in range(0, len(ecg_flat), chunk_size):
    # Extract chunk
    chunk = ecg_flat[i:i + chunk_size] #提取chunk_size大小的数据，从i到i+8

    # Convert chunk to byte
    chunk_bytes = chunk.tobytes() #转化为bytes

    # Fill data to 32-byte alignment
    remainder = len(chunk_bytes) % 32
    if remainder != 0:
        padding_length = 32 - remainder
        chunk_bytes += b'\x00' * padding_length

    # encrypted data block
    ciphertext = Kyber512._cpapke_enc(public_key, chunk_bytes, coins=os.urandom(32))

    # Save ciphertext
    ciphertexts.append(ciphertext)

# encryption time
encryption_end_time = time.time()
encryption_elapsed_time = encryption_end_time - encryption_start_time
print(f"Encryption completed：{encryption_elapsed_time:.4f} seconds")

#尝试画加密后的图
encrypted_ecg = np.array([int.from_bytes(ct[:8], byteorder='little') for ct in ciphertexts])

# Save the ciphertext as a .npy file
np.save(encrypted_file, np.array(ciphertexts, dtype=object))
print(f"\nThe ciphertext has been saved to：{encrypted_file}")

# ========== Decrypting ECG Data ==========
print("\nStart decrypting ECG signals...")

decrypted_chunks = []

# Record decryption time
decryption_start_time = time.time()

for ciphertext in ciphertexts:
    # decrypted data block
    decrypted_chunk_bytes = Kyber512._cpapke_dec(secret_key, ciphertext)

    # Remove Fill
    decrypted_chunk_bytes = decrypted_chunk_bytes[:len(chunk.tobytes())]

    # Transfer decrypted bytes back to NumPy array
    decrypted_chunk = np.frombuffer(decrypted_chunk_bytes, dtype=np.float32)

    # Save decryption results
    decrypted_chunks.append(decrypted_chunk)

# Decryption time
decryption_end_time = time.time()
decryption_elapsed_time = decryption_end_time - decryption_start_time
print(f"Decryption completed,took：{decryption_elapsed_time:.4f} seconds")

# Combining decrypted data
decrypted_ecg = np.concatenate(decrypted_chunks)

# Save decrypted data as a .npy file
np.save(decrypted_file, decrypted_ecg)
print(f"\nThe decrypted ECG data has been saved to the：{decrypted_file}")

# ========== Visualization Comparison ==========
# print("\nCompare and contrast the original signal with the decrypted signal...")
#
# # Adjust original and decrypted signals to the same length
# decrypted_ecg = decrypted_ecg[:len(ecg_flat)]
#
# plt.figure(figsize=(12, 8))
#
# plt.subplot(3, 1, 1)
# plt.title("Original ECG")
# plt.plot(ecg_flat, label="Original ECG", color='blue')
# plt.legend()
#
# plt.subplot(3, 1, 2)
# plt.title("Encrypted ECG (Converted to int for visualization)")
# plt.plot(encrypted_ecg, label="Encrypted ECG", color='red')
# plt.legend()
#
# plt.subplot(3, 1, 3)
# plt.title("Decrypted ECG")
# plt.plot(decrypted_ecg, label="Decrypted ECG", color='green')
# plt.legend()

# plt.tight_layout()
# plt.show()


print("\n========== Encryption Effect Evaluation ==========")

# 1.  计算香农熵（Shannon Entropy）
def compute_entropy(data):
    # 将数据归一化为整数频率分布（适合float类型）
    data = np.round(data, decimals=4)  # 保留小数精度，防止唯一值太多
    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

entropy_original = compute_entropy(ecg_flat)
entropy_encrypted = compute_entropy(np.array([int.from_bytes(ct[:8], 'little') for ct in ciphertexts]))

print(f"Shannon Entropy of Original ECG  : {entropy_original:.4f}")
print(f"Shannon Entropy of Encrypted ECG : {entropy_encrypted:.4f}")

# 2.  计算原始信号 vs 加密信号的皮尔逊相关性
# 将加密信号转为整数
encrypted_ints = np.array([int.from_bytes(ct[:8], 'little') for ct in ciphertexts])

# 匹配维度（因为原始信号比加密信号长）
min_len = min(len(encrypted_ints), len(ecg_flat))
correlation, _ = pearsonr(ecg_flat[:min_len], encrypted_ints[:min_len])

print(f"Pearson Correlation (Original vs Encrypted) : {correlation:.4f}")

# ========== 直方图对比 ==========
import matplotlib.pyplot as plt

# 转换加密数据为整数序列（只用前 8 字节转换为整数）
encrypted_ints = np.array([int.from_bytes(ct[:8], 'little') for ct in ciphertexts])

# 画图
plt.figure(figsize=(12, 5))

# 原始ECG直方图
plt.subplot(1, 2, 1)
plt.hist(ecg_flat, bins=100, color='blue', alpha=0.7)
plt.title("Histogram of Original ECG")
plt.xlabel("Signal Value")
plt.ylabel("Frequency")

# 加密ECG直方图
plt.subplot(1, 2, 2)
plt.hist(encrypted_ints, bins=100, color='red', alpha=0.7)
plt.title("Histogram of Encrypted ECG (int converted)")
plt.xlabel("Integer Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# # 设置抽样间隔
# sample_interval = 100  # 你用的是 1000 就写 1000
#
# # 生成绘图用的下采样数据
# ecg_flat_plot = ecg_flat[::sample_interval]
# decrypted_ecg_plot = decrypted_ecg[::sample_interval]
#
# # 加密数据转换为整数趋势
# encrypted_ecg_full = np.array([int.from_bytes(ct[:8], 'little') for ct in ciphertexts])
#
# # 计算加密数据在原始数组中对应的位置（一个 chunk 代表 8 个原始点）
# encrypted_indices = np.arange(len(encrypted_ecg_full)) * chunk_size
#
# # 下采样这些加密点的位置，使其对应下采样的原始数据
# encrypted_indices_plot = encrypted_indices[::sample_interval]
#
# # 再取对应的加密值
# encrypted_ecg_plot = encrypted_ecg_full[::sample_interval]
#
# # 绘图
# plt.figure(figsize=(12, 8))
#
# plt.subplot(3, 1, 1)
# plt.title(f"Original ECG (every {sample_interval} points)")
# plt.plot(ecg_flat_plot, label="Original ECG", color='blue')
# plt.legend()
#
# plt.subplot(3, 1, 2)
# plt.title("Encrypted ECG (converted to int)")
# plt.plot(encrypted_indices_plot, encrypted_ecg_plot, label="Encrypted ECG", color='red')
# plt.legend()
#
# plt.subplot(3, 1, 3)
# plt.title(f"Decrypted ECG (every {sample_interval} points)")
# plt.plot(decrypted_ecg_plot, label="Decrypted ECG", color='green')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

# Verify the decryption result
# if np.allclose(ecg_flat, decrypted_ecg, atol=1e-5):
#     print("\nDecryption successful! The original signal agrees with the decrypted signal!")
# else:
#     print("\nDecryption failed! The original signal does not match the decrypted signal!")