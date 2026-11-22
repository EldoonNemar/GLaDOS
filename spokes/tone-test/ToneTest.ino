/*
 * ToneTest.ino v2 — XVF3800 speaker power probe, full-scale + optional wifi
 *
 * Plays a random sine tone (200-1000 Hz, 500 ms) every 3 seconds straight
 * out of the XVF3800's speaker amp. NO hub, NO mic — isolates the
 * TX -> DAC -> amp -> speaker -> power path.
 *
 * TONE_AMP_PCT : amplitude percent. v1 ran at 25; the B104 death used
 *                near-full-scale TTS (~13x the power), so v2 defaults to
 *                100 to reproduce those electrical conditions.
 * TONE_WIFI    : 0 = no wifi (pure analog test, v1 behavior)
 *                1 = also connect wifi, reproducing B104's full condition
 *                    (I2S TX + wifi together)
 *
 * Build: Arduino IDE, board = Seeed XIAO ESP32S3, USB CDC On Boot = Enabled.
 *
 * Watch for:
 *   - tones + stable            -> analog path fine at full scale
 *   - dies on a tone             -> current trip; power/load problem,
 *                                    independent of the spoke firmware
 *   - TONE_WIFI=1 dies, =0 fine  -> the wifi+amp combination is what
 *                                    trips the supply
 */
#include "driver/i2s_std.h"
#include <math.h>

#define TONE_AMP_PCT 100
#define TONE_WIFI    0

#if TONE_WIFI
#include "esp_wifi.h"
#include "esp_event.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#define SSID     "1.21Jigowatts"
#define PASSWORD "Jigowatts"
static EventGroupHandle_t wifi_eg;
static void wifi_handler(void *arg, esp_event_base_t b, int32_t id, void *d) {
    if (id == WIFI_EVENT_STA_START) esp_wifi_connect();
    if (id == IP_EVENT_STA_GOT_IP) xEventGroupSetBits(wifi_eg, 1);
}
#endif

#define I2S_BCLK ((gpio_num_t)8)
#define I2S_WS   ((gpio_num_t)7)
#define I2S_DOUT ((gpio_num_t)44)

#define SAMPLE_RATE 48000
#define TONE_MS     500
#define GAP_MS      3000

static i2s_chan_handle_t tx;
static int32_t chunk[1000 * 2];  // 8 KB static — the loop() task stack is small

static void i2s_tx_init() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_SLAVE);
    chan_cfg.auto_clear = true;
    i2s_new_channel(&chan_cfg, &tx, NULL);  // TX only
    i2s_std_config_t cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,   // XVF generates MCLK internally
            .bclk = I2S_BCLK,
            .ws   = I2S_WS,
            .dout = I2S_DOUT,
            .din  = I2S_GPIO_UNUSED,
            .invert_flags = {0},
        },
    };
    i2s_channel_init_std_mode(tx, &cfg);
    i2s_channel_enable(tx);
}

static void play_tone(uint32_t hz) {
    const long amp = (long)(32767.0 * TONE_AMP_PCT / 100.0);
    const size_t total = (size_t)SAMPLE_RATE * TONE_MS / 1000;  // 24000 frames
    for (size_t start = 0; start < total; start += 1000) {
        const size_t n = min((size_t)1000, total - start);
        for (size_t i = 0; i < n; i++) {
            int32_t s = (int32_t)((long)sin(2.0 * M_PI * (double)hz *
                            (double)(start + i) / (double)SAMPLE_RATE) * amp) << 16;
            chunk[i * 2]     = s;
            chunk[i * 2 + 1] = s;
        }
        size_t to_write = n * 8, written = 0;
        while (written < to_write) {
            size_t w = 0;
            esp_err_t r = i2s_channel_write(tx, chunk + (written / 4),
                                            to_write - written, &w, pdMS_TO_TICKS(500));
            if (r != ESP_OK || w == 0) break;
            written += w;
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);
    Serial.printf("ToneTest v2 — amp=%d%% wifi=%d\n", TONE_AMP_PCT, (int)TONE_WIFI);

#if TONE_WIFI
    WiFi.mode(WIFI_STA);
    wifi_eg = xEventGroupCreate();
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_handler, nullptr, nullptr);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_handler, nullptr, nullptr);
    WiFi.setHostname("tone-test");
    WiFi.begin(SSID, PASSWORD);
    if (xEventGroupWaitBits(wifi_eg, 1, false, false, pdMS_TO_TICKS(10000)))
        Serial.printf("wifi up: %s\n", WiFi.localIP().toString().c_str());
    else
        Serial.println("wifi: no IP in 10 s, continuing anyway");
#endif

    i2s_tx_init();
    Serial.println("I2S TX up (48k/32/stereo, slave); tones every 3 s");
    randomSeed(esp_random());
}

void loop() {
    uint32_t hz = 200 + random(801);  // 200-1000 Hz
    Serial.printf("tone %u Hz\n", hz);
    play_tone(hz);
    delay(GAP_MS);
}
