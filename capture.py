import RPi.GPIO as GPIO
import subprocess
import requests
import time

RED_LED   = 17
WHITE_LED = 27
LAPTOP_IP = "192.168.1.24"

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED,   GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(WHITE_LED, GPIO.OUT, initial=GPIO.LOW)

print('=== Nayana Eye Screening Device ===')
print('Press ENTER to capture | Type q then ENTER to quit')

count = 1
while True:
    cmd = input('\n> Press ENTER to capture (or q to quit): ')
    if cmd.strip().lower() == 'q':
        print('Exiting...')
        break

    OUTPUT = f'/home/nayana/eye_{count}.jpg'

    # Red blinks for focus
    print('Look at the red light...')
    for _ in range(5):
        GPIO.output(RED_LED, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(RED_LED, GPIO.LOW)
        time.sleep(0.2)

    # Both on for capture
    print(f'Capturing image {count}...')
    GPIO.output(RED_LED,   GPIO.HIGH)
    GPIO.output(WHITE_LED, GPIO.HIGH)
    time.sleep(0.5)

    subprocess.run(['rpicam-still', '-o', OUTPUT,
                    '--width', '1920', '--height', '1080',
                    '--timeout', '2000'])

    GPIO.output(RED_LED,   GPIO.LOW)
    GPIO.output(WHITE_LED, GPIO.LOW)
    print(f'Saved: {OUTPUT}')

    # Send to Nayana
    print('Sending to Nayana...')
    try:
        with open(OUTPUT, 'rb') as f:
            r = requests.post(f'http://{LAPTOP_IP}:5000/upload',
                              files={'image': (f'eye_{count}.jpg', f, 'image/jpeg')},
                              timeout=15)
        if r.status_code == 200:
            print(f'Sent successfully!')
        else:
            print('Send failed:', r.status_code)
    except Exception as e:
        print('Error sending:', e)

    count += 1

GPIO.cleanup()
print('Done.')
