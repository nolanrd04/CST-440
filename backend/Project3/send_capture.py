#!/usr/bin/env python3
"""直接发送'c'命令给Arduino并等待响应"""
import serial
import time
import sys

port = '/dev/cu.usbmodem1101'
print(f"连接到 {port}...")

try:
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)  # 等待Arduino准备好
    
    print("✓ 已连接")
    print("\n📸 发送捕获命令...")
    ser.write(b'c\n')  # 发送'c'加换行符
    ser.flush()
    
    print("\n=== 等待Arduino响应（最多30秒）===\n")
    
    start_time = time.time()
    got_response = False
    
    while time.time() - start_time < 30:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
                got_response = True
                
                # 如果看到完整结果，退出
                if 'FACE DETECTED' in line or 'NO FACE' in line:
                    print("\n✓ 检测完成")
                    break
        time.sleep(0.01)
    
    if not got_response:
        print("\n⚠️  30秒内没有收到Arduino响应")
        print("可能的问题：")
        print("  1. 相机硬件未正确连接")
        print("  2. Arduino程序卡住")
        print("  3. 需要重启Arduino")
    
    ser.close()
    
except serial.SerialException as e:
    print(f"串口错误: {e}")
    print(f"\n检查Arduino是否连接：")
    import subprocess
    result = subprocess.run(['ls', '/dev/cu.*'], capture_output=True, text=True, shell=True)
    print(result.stdout)
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n用户中断")
    ser.close()
