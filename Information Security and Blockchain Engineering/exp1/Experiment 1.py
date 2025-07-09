from flask import Flask, request, jsonify, render_template
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import os

app = Flask(__name__, static_folder='static')


def parse_ascii_to_utf8(input_string):
    # 将输入的 ASCII 码字符串转换为字节序列
    byte_sequence = bytes.fromhex(input_string.replace(" ", ""))
    # 将字节序列解码为 UTF-8 编码的字符串
    utf8_string = byte_sequence.decode('utf-8')
    return utf8_string


def sha256_hash(input_string):
    # 将 ASCII 码格式的输入文本转换为 UTF-8 编码的字符串
    utf8_string = parse_ascii_to_utf8(input_string)
    # 创建一个新的 SHA-256 哈希对象
    sha256 = hashlib.sha256()
    # 将 UTF-8 编码的字符串进行更新
    sha256.update(utf8_string.encode('utf-8'))
    # 返回哈希值的十六进制字符串表示
    return sha256.hexdigest()


def aes_encrypt(input_string, key):
    # 使用 SHA-256 从密钥生成一个 256 位的密钥
    key = hashlib.sha256(key.encode()).digest()
    # 生成一个随机的初始化向量 (IV)
    iv = os.urandom(16)
    # 创建一个新的 AES 密码实例
    cipher = AES.new(key, AES.MODE_CBC, iv)
    # 加密 UTF-8 编码的字符串，并填充至块大小
    encrypted_bytes = cipher.encrypt(pad(input_string.encode('utf-8'), AES.block_size))
    # 对加密的数据（包括 IV）进行 Base64 编码，并转换为 UTF-8 字符串
    encrypted_string = base64.b64encode(iv + encrypted_bytes).decode('utf-8')
    return encrypted_string


def aes_decrypt(encrypted_string, key):
    # 使用 SHA-256 从密钥生成一个 256 位的密钥
    key = hashlib.sha256(key.encode()).digest()
    # 对加密字符串进行 Base64 解码
    encrypted_data = base64.b64decode(encrypted_string)
    # 提取初始化向量 (IV)
    iv = encrypted_data[:16]
    # 提取加密后的数据
    encrypted_bytes = encrypted_data[16:]
    # 创建一个新的 AES 密码实例
    cipher = AES.new(key, AES.MODE_CBC, iv)
    # 解密数据并去除填充
    decrypted_string = unpad(cipher.decrypt(encrypted_bytes), AES.block_size).decode('utf-8')
    return decrypted_string


@app.route('/')
def index():
    # 渲染首页
    return render_template('index.html')


@app.route('/sha256', methods=['POST'])
def handle_sha256():
    # 获取请求中的输入字符串
    input_string = request.json.get('input_string')
    # 计算 SHA-256 哈希值
    hash_value = sha256_hash(input_string)
    # 返回 JSON 格式的哈希值
    return jsonify({"hash_value": hash_value})


@app.route('/encrypt', methods=['POST'])
def handle_encrypt():
    # 获取请求中的输入字符串和密钥
    input_string = request.json.get('input_string')
    key = request.json.get('key')
    # 进行 AES 加密
    encrypted_string = aes_encrypt(input_string, key)
    # 返回 JSON 格式的加密字符串
    return jsonify({"encrypted_string": encrypted_string})


@app.route('/decrypt', methods=['POST'])
def handle_decrypt():
    # 获取请求中的加密字符串和密钥
    encrypted_string = request.json.get('encrypted_string')
    key = request.json.get('key')
    # 进行 AES 解密
    decrypted_string = aes_decrypt(encrypted_string, key)
    # 返回 JSON 格式的解密字符串
    return jsonify({"decrypted_string": decrypted_string})


if __name__ == '__main__':
    app.run(debug=True)
