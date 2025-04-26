import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kyber import Kyber512
import os, time
from scipy.stats import entropy, pearsonr
import wfdb
import ast
# parameters
csv_path = 'PTB_ANLS/ptbxl_database.csv'
record_base = 'PTB_ANLS/records500/00000'
target_label = 'HYP'
lead_index = 1  # Lead II
num_records = 3
chunk_size = 8
root_dir = 'PTB_ANLS'
df = pd.read_csv(csv_path)

# From String to Dictionary
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
df_map = pd.read_csv('PTB_ANLS/scp_statements.csv')

# Keep only diagnostic type labels
df_map = pd.read_csv('PTB_ANLS/scp_statements.csv', encoding='utf-8-sig')
df_map = df_map[df_map['diagnostic_class'].notna()]
# Creating a mapping table（scp_code -> superclass）
code_to_class = df_map.set_index('scp_code')['diagnostic_class'].to_dict()
# 提取每条记录的标签大类（可能有多个，我们取第一个为主标签）
def extract_superclass(scp_dict):
    classes = [code_to_class.get(code) for code in scp_dict.keys() if code in code_to_class]
    return classes[0] if classes else 'OTHER'

df['diagnostic_superclass'] = df['scp_codes'].apply(extract_superclass)

# 筛选包含 'NORM' 的记录
df = df[df['diagnostic_superclass'] == 'HYP']

# 提取前3条文件路径
file_list = df['filename_hr'].values[:3]

# === 读取信号并拼接 ===
all_ecg = []
for filename in file_list:
    record_path = os.path.join(root_dir, filename)  # 正确拼接完整路径
    print("读取路径：", record_path)
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal[:, lead_index]  # 取 Lead II
    all_ecg.append(sig.flatten())

ecg_flat = np.concatenate(all_ecg).astype(np.float32)

# === Kyber 密钥生成 ===
public_key, secret_key = Kyber512.keygen()

# === 加密 ===
ciphertexts = []
encryption_start_time = time.time()
for i in range(0, len(ecg_flat), chunk_size):
    chunk = ecg_flat[i:i + chunk_size]
    chunk_bytes = chunk.tobytes()
    remainder = len(chunk_bytes) % 32
    if remainder != 0:
        chunk_bytes += b'\x00' * (32 - remainder)
    ciphertext = Kyber512._cpapke_enc(public_key, chunk_bytes, coins=os.urandom(32))
    ciphertexts.append(ciphertext)
encryption_end_time = time.time()

# === 解密 ===
decrypted_chunks = []
for i, ciphertext in enumerate(ciphertexts):
    decrypted_chunk_bytes = Kyber512._cpapke_dec(secret_key, ciphertext)
    decrypted_chunk_bytes = decrypted_chunk_bytes[:chunk_size * 4]
    decrypted_chunk = np.frombuffer(decrypted_chunk_bytes, dtype=np.float32)
    decrypted_chunks.append(decrypted_chunk)
decrypted_ecg = np.concatenate(decrypted_chunks)
decrypted_ecg = decrypted_ecg[:len(ecg_flat)]

# === 转换密文为整数用于直方图展示
encrypted_ints = np.array([int.from_bytes(ct[:8], 'little') for ct in ciphertexts])


# === 熵和相关性分析 ===
def compute_entropy(data):
    data = np.round(data, decimals=4)
    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)


entropy_orig = compute_entropy(ecg_flat)
entropy_encrypted = compute_entropy(encrypted_ints)
correlation, _ = pearsonr(ecg_flat[:len(encrypted_ints)], encrypted_ints)

print("\n========== 加密效果评估 ==========")
print(f"Shannon Entropy of Original ECG  : {entropy_orig:.4f}")
print(f"Shannon Entropy of Encrypted ECG : {entropy_encrypted:.4f}")
print(f"Pearson Correlation (Original vs Encrypted) : {correlation:.4f}")

# # === 画图对比 ===
# plt.figure(figsize=(12, 6))
#
# plt.subplot(3, 1, 1)
# plt.title("Original ECG (Lead II)")
# plt.plot(ecg_flat, color='blue')
#
# plt.subplot(3, 1, 2)
# plt.title("Encrypted ECG (int converted)")
# plt.plot(encrypted_ints, color='red')
#
# plt.subplot(3, 1, 3)
# plt.title("Decrypted ECG (Lead II)")
# plt.plot(decrypted_ecg, color='green')
#
# plt.tight_layout()
# plt.show()

# === 直方图对比 ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(ecg_flat, bins=100, color='blue', alpha=0.7)
plt.title("Original ECG Histogram")

plt.subplot(1, 2, 2)
plt.hist(encrypted_ints, bins=100, color='red', alpha=0.7)
plt.title("Encrypted ECG Histogram (converted ints)")

plt.tight_layout()
plt.show()