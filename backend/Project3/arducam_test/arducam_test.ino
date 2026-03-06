/*
  ArduCAM OV2640 Hardware Test
  Tests SPI communication and camera initialization
*/

#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>

#define CS_PIN 10
ArduCAM cam(OV2640, CS_PIN);

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  
  Serial.println("\n=== ArduCAM Hardware Test ===\n");
  
  // Test 1: Initialize I2C and SPI
  Serial.println("Test 1: Initializing I2C and SPI...");
  Wire.begin();
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  Serial.println("  ✓ I2C and SPI initialized");
  
  // Test 2: Verify SPI communication with ArduCAM
  Serial.println("\nTest 2: Testing SPI communication...");
  cam.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t test_val = cam.read_reg(ARDUCHIP_TEST1);
  
  if (test_val == 0x55) {
    Serial.println("  ✓ SPI communication OK");
    Serial.print("  Test register value: 0x");
    Serial.println(test_val, HEX);
  } else {
    Serial.println("  ✗ SPI communication FAILED!");
    Serial.print("  Expected: 0x55, Got: 0x");
    Serial.println(test_val, HEX);
    Serial.println("\n=== TEST FAILED ===");
    Serial.println("Check wiring:");
    Serial.println("  - CS  -> Pin 10");
    Serial.println("  - MOSI -> Pin 11");
    Serial.println("  - MISO -> Pin 12");
    Serial.println("  - SCK  -> Pin 13");
    while(1) { delay(1000); }
  }
  
  // Test 3: Verify OV2640 camera chip ID
  Serial.println("\nTest 3: Reading OV2640 chip ID...");
  uint8_t vid, pid;
  cam.wrSensorReg8_8(0xff, 0x01);  // Select sensor register bank
  cam.rdSensorReg8_8(0x0A, &vid);  // Read chip ID high byte
  cam.rdSensorReg8_8(0x0B, &pid);  // Read chip ID low byte
  
  Serial.print("  Chip ID: 0x");
  Serial.println((vid << 8) | pid, HEX);
  
  if (vid == 0x26) {
    Serial.println("  ✓ OV2640 camera detected");
  } else {
    Serial.println("  ✗ OV2640 NOT detected!");
    Serial.println("\n=== TEST FAILED ===");
    Serial.println("Check camera module:");
    Serial.println("  - VCC -> 3.3V or 5V (check your module)");
    Serial.println("  - GND -> GND");
    Serial.println("  - SDA -> A4");
    Serial.println("  - SCL -> A5");
    while(1) { delay(1000); }
  }
  
  // Test 4: Initialize camera
  Serial.println("\nTest 4: Initializing camera...");
  cam.set_format(JPEG);
  cam.InitCAM();
  cam.set_bit(ARDUCHIP_TIM, VSYNC_MASK);
  cam.clear_fifo_flag();
  Serial.println("  ✓ Camera initialized");
  
  // Test 5: Switch to RGB565 mode
  Serial.println("\nTest 5: Setting RGB565 mode...");
  cam.wrSensorReg8_8(0xFF, 0x00);  // Select DSP register bank
  cam.wrSensorReg8_8(0xDA, 0x08);  // IMAGE_MODE: non-compressed RGB565
  Serial.println("  ✓ RGB565 mode set");
  
  // Test 6: Test image capture
  Serial.println("\nTest 6: Testing image capture...");
  cam.flush_fifo();
  cam.clear_fifo_flag();
  cam.start_capture();
  
  Serial.println("  Waiting for capture...");
  unsigned long start = millis();
  while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
    if (millis() - start > 5000) {
      Serial.println("  ✗ Capture timeout (5 seconds)");
      Serial.println("\n=== TEST FAILED ===");
      Serial.println("Camera is not capturing images.");
      Serial.println("Possible issues:");
      Serial.println("  - Camera module power supply insufficient");
      Serial.println("  - Camera lens not installed");
      Serial.println("  - Faulty camera module");
      while(1) { delay(1000); }
    }
    delay(1);
  }
  
  uint32_t fifo_len = cam.read_fifo_length();
  Serial.print("  ✓ Capture complete! FIFO length: ");
  Serial.print(fifo_len);
  Serial.println(" bytes");
  
  if (fifo_len > 0) {
    Serial.println("  Image data is available in FIFO");
  } else {
    Serial.println("  ✗ FIFO is empty!");
  }
  
  Serial.println("\n=== ALL TESTS PASSED ===");
  Serial.println("ArduCAM is working correctly!");
  Serial.println("\nPress 'c' to capture a test frame.");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c') {
      Serial.println("\n--- Capturing test frame ---");
      
      cam.flush_fifo();
      cam.clear_fifo_flag();
      cam.start_capture();
      
      unsigned long start = millis();
      while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
        if (millis() - start > 5000) {
          Serial.println("Capture timeout!");
          return;
        }
        delay(1);
      }
      
      uint32_t fifo_len = cam.read_fifo_length();
      Serial.print("FIFO length: ");
      Serial.print(fifo_len);
      Serial.println(" bytes");
      
      // Read first 20 bytes as sample
      cam.CS_LOW();
      SPI.transfer(BURST_FIFO_READ);
      Serial.print("First 20 bytes: ");
      for (int i = 0; i < 20; i++) {
        uint8_t b = SPI.transfer(0x00);
        if (b < 16) Serial.print("0");
        Serial.print(b, HEX);
        Serial.print(" ");
      }
      cam.CS_HIGH();
      Serial.println("\n✓ Test capture complete");
    }
  }
}
