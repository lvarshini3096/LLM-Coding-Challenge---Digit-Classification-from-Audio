import sounddevice as sd

def check_microphone_settings():
    """
    Queries and displays information about available audio devices,
    highlighting the default input device.
    """
    print("--- Querying Audio Devices ---")
    try:
        # Print all devices for a complete overview
        print("Full device list:")
        print(sd.query_devices())
        
        # Get information specifically about the default input device
        default_input_device_info = sd.query_devices(kind='input')
        
        print("\n--- Default Input Device ---")
        if default_input_device_info and 'name' in default_input_device_info:
            print(f"Name: {default_input_device_info['name']}")
            print(f"Max Input Channels: {default_input_device_info['max_input_channels']}")
            print(f"Default Sample Rate: {default_input_device_info['default_samplerate']} Hz")
            
            if default_input_device_info['max_input_channels'] > 0:
                print("\n It looks like a default input device is configured correctly.")
            else:
                print("\n Warning: The default device does not seem to be a valid input device (microphone).")
        else:
            print("\n Error: Could not find a default input device.")
            print("Please check your system's sound settings and ensure a microphone is connected and selected as the default.")

    except Exception as e:
        print(f"\nAn error occurred while querying devices: {e}")

if __name__ == "__main__":
    check_microphone_settings()

