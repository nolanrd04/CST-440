#!/usr/bin/env python3
"""
测试人脸检测程序
"""
import serial
import time

port = '/dev/cu.usbmodem1101'

print("🔄 连接到Arduino...")
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(3)  # 等待Arduino完全启动

print("📖 读取启动信息...")
for _ in range(30):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
    time.sleep(0.1)

print("\n" + "="*60)
print("📸 发送捕获命令 'c'")
print("="*60 + "\n")

ser.write(b'c\n')
ser.flush()

print("⏳ 等待响应（最多30秒）...\n")
start_time = time.time()
got_response = False

while time.time() - start_time < 30:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
            got_response = True
            
            if 'FACE DETECTED' in line or 'NO FACE' in line:
                print("\n✅ 捕获完成！")
                break
            if 'DEBUG_IMAGE_END' in line:
                print("\n✅ 调试图像接收完成！")
                break
    time.sleep(0.01)

if not got_response:
    print("\n❌ 没有收到响应")
    print("可能的问题：")
    print("  1. 程序没有正确接收 'c' 命令")
    print("  2. 程序卡在某个地方")
    print("  3. 相机捕获耗时过长")
else:
    print("\n✅ 程序正常工作！")

ser.close()
