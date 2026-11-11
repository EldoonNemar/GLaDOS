idf_component_register(
    SRC_DIRS "."
    INCLUDE_DIRS "."
    REQUIRES driver nvs_flash esp_wifi esp_event esp_netif lwip mqtt
    PRIV_REQUIRES json
)

# KConfig defaults
target_compile_options(${COMPONENT_TARGET} PRIVATE
    -DCONFIG_WIFI_SSID='"your_wifi_ssid"'
    -DCONFIG_WIFI_PASSWORD='"your_wifi_password"'
    -DCONFIG_HUB_HOST='"localhost"'
    -DCONFIG_HUB_PORT=5051
)
