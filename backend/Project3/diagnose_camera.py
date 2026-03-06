#!/usr/bin/env python3
"""
ArduCAM诊断工具
监听Arduino串口输出，分析相机初始化过程
"""
import serial
import time
import sys

def find_arduino():
    """查找Arduino端口"""
    import glob
    ports = glob.glob('/dev/cu.usbmodem*')
    if not ports:
        ports = glob.glob('/dev/ttyACM*')
    return ports[0] if ports else None

def diagnose_camera():
    port = find_arduino()
    if not port:
        print("❌ 未找到Arduino设备")
        print("请检查：")
        print("  1. Arduino是否通过USB连接")
        print("  2. USB线是否正常")
        return False
    
    print(f"📱 连接到: {port}\n")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # 等待Arduino重启
        
        print("=== 读取Arduino初始化信息 ===\n")
        
        tests_passed = {
            'spi': False,
            'i2c': False,
            'chip_id': False,
            'init': False,
            'capture': False
        }
        
        chip_id = None
        fifo_length = None
        
        start_time = time.time()
        while time.time() - start_time < 10:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)
                    
                    # 分析输出
                    if 'SPI communication OK' in line or 'SPI' in line and '✓' in line:
                        tests_passed['spi'] = True
                    if 'I2C' in line and '✓' in line:
                        tests_passed['i2c'] = True
                    if 'OV2640 chip ID' in line or 'Chip ID' in line:
                        tests_passed['chip_id'] = True
                        # 提取芯片ID
                        if '0x' in line:
                            chip_id = line.split('0x')[1].strip()
                    if 'Camera initialized' in line or 'Camera ready' in line:
                        tests_passed['init'] = True
                    if 'TFLite ready' in line or 'ALL TESTS PASSED' in line:
                        tests_passed['capture'] = True
                    if 'FIFO length' in line:
                        try:
                            fifo_length = int(line.split(':')[1].split()[0])
                        except:
                            pass
            time.sleep(0.01)
        
        print("\n" + "="*50)
        print("📊 诊断结果:")
        print("="*50)
        
        print(f"{'SPI通信:':<20} {'✓' if tests_passed['spi'] else '✗'}")
        print(f"{'I2C通信:':<20} {'✓' if tests_passed['i2c'] else '✗'}")
        print(f"{'相机芯片检测:':<20} {'✓' if tests_passed['chip_id'] else '✗'}")
        if chip_id:
            expected = chip_id.startswith('26')
            print(f"  芯片ID: {chip_id} {'(正确)' if expected else '(错误!应该是0x26xx)'}")
        print(f"{'相机初始化:':<20} {'✓' if tests_passed['init'] else '✗'}")
        print(f"{'图像捕获:':<20} {'✓' if tests_passed['capture'] else '✗'}")
        if fifo_length is not None:
            print(f"  FIFO大小: {fifo_length} bytes")
        
        print("\n" + "="*50)
        
        # 总体评估
        all_passed = all(tests_passed.values())
        if all_passed:
            print("✅ 相机模块工作正常！")
            return True
        else:
            print("❌ 相机模块有问题")
            print("\n🔧 故障排除建议:")
            
            if not tests_passed['spi']:
                print("\n【SPI通信失败】")
                print("检查接线：")
                print("  CS   -> Pin 10")
                print("  MOSI -> Pin 11") 
                print("  MISO -> Pin 12")
                print("  SCK  -> Pin 13")
                
            if not tests_passed['chip_id']:
                print("\n【相机芯片未检测到】")
                print("检查接线：")
                print("  SDA -> A4")
                print("  SCL -> A5")
                print("  VCC -> 3.3V 或 5V")
                print("  GND -> GND")
                print("\n可能原因：")
                print("  - I2C连接松动")
                print("  - 相机模块损坏")
                print("  - 电源不足")
                
            if not tests_passed['capture']:
                print("\n【图像捕获失败】")
                print("可能原因：")
                print("  - 镜头未安装或松动")
                print("  - 电源电流不足")
                print("  - 相机模块故障")
            
            return False
        
    except serial.SerialException as e:
        print(f"❌ 串口错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return False
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == '__main__':
    print("🔍 ArduCAM诊断工具\n")
    print("请确保已上传测试代码到Arduino")
    print("(使用 arducam_test.ino 或 face_detector_arduino.ino)\n")
    
    input("按回车开始诊断...")
    diagnose_camera()
