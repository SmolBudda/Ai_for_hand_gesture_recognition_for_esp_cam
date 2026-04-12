#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>

// Konfiguracja WiFi
const char* SSID = "Tel-AWiFi";          // Zmień na swoją sieć WiFi
const char* PASSWORD = "SlalomAlejkom";  // Zmień na hasło WiFi

// Pinout dla ESP32-CAM (AI-Thinker)
#define PWDN_GPIO_NUM    32
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

WebServer server(80);

// Inicjalizacja kamery
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;  // 640x480
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Błąd inicjalizacji kamery: 0x%x\n", err);
    return;
  }
  Serial.println("Kamera zainicjalizowana poprawnie!");

  // Ustawienia kamery
  sensor_t * s = esp_camera_sensor_get();
  s->set_brightness(s, 0);     // -2 to 2
  s->set_contrast(s, 0);       // -2 to 2
  s->set_saturation(s, 0);     // -2 to 2
  s->set_special_effect(s, 0); // 0 to 6 (no effect, grayscale, sepia, etc)
  s->set_whitebal(s, 1);       // auto white balance
}

// Strona HTML
void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>ESP32-CAM Stream</title>
    <style>
        body { font-family: Arial; text-align: center; background: #f0f0f0; }
        .container { max-width: 800px; margin: 50px auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        img { max-width: 100%; height: auto; border: 2px solid #333; border-radius: 5px; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 ESP32-CAM Live Stream</h1>
        <img src="http://)" + String(WiFi.localIP().toString()) + R"(:/stream" alt="Camera Stream" width="640" />
    </div>
</body>
</html>
  )";
  server.send(200, "text/html", html);
}

// Stream MJPEG
void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n";
  response += "Connection: keep-alive\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
      delay(1);
      continue;
    }

    client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("\n\n--- ESP32-CAM inicjalizacja ---");

  // Inicjalizacja kamery
  initCamera();

  // Połączenie WiFi
  Serial.print("Łączenie z WiFi: ");
  Serial.println(SSID);
  WiFi.begin(SSID, PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nPołączona!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nNie udało się połączyć z WiFi!");
  }

  // Konfiguracja serwera HTTP
  server.on("/", handleRoot);
  server.on("/stream", HTTP_GET, handleStream);
  server.begin();
  Serial.println("Serwer HTTP uruchomiony!");
  Serial.print("Otwórz: http://");
  Serial.println(WiFi.localIP());
}

void loop() {
  server.handleClient();
  delay(1);
} 