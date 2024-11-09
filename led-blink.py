import time

# Path to the ACT LED's brightness control
led_path = "/sys/class/leds/ACT/brightness"

def set_led(state):
    """Set LED state. 1 for on, 0 for off."""
    with open(led_path, 'w') as led:
        led.write(str(state))

try:
    while True:
        set_led(1)     # Turn LED on
        time.sleep(0.5)
        set_led(0)     # Turn LED off
        time.sleep(0.5)
except KeyboardInterrupt:
    set_led(0)         # Ensure LED is off when exiting
    print("Exiting script and turning LED off.")


