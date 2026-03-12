import serial
import time
import sys

# Mogelijke poorten op de Raspberry Pi
ports = ['/dev/serial0', '/dev/ttyS0', '/dev/ttyAMA0', '/dev/ttyUSB0']

def test_connection():
    ser = None
    selected_port = None

    print("--- UART Test Script ---")
    
    # 1. Probeer poort te openen
    for port in ports:
        try:
            print(f"Proberen te verbinden met {port}...")
            ser = serial.Serial(port, 115200, timeout=1)
            selected_port = port
            print(f"--> SUCCES! Verbonden met {port}")
            break
        except Exception as e:
            print(f"   (Mislukt: {e})")
    
    if not ser:
        print("\nFOUT: Kan geen enkele seriële poort openen.")
        print("Tips:")
        print("1. Heb je 'Serial Port' aan staan in raspi-config?")
        print("2. Is de 'Serial Console' UIT in raspi-config?")
        return

    # 2. Luisteren en Zenden
    print(f"\nStart test op {selected_port}. Druk op CTRL+C om te stoppen.")
    print("Stuur commando's: 'pmd', 'rest', 'karton', 'organisch', 'hit', 'off'")
    
    try:
        # Wacht even op reset van ESP (soms nodig bij openen poort)
        time.sleep(2) 
        
        while True:
            # Stuur een test commando
            cmd = input("\nTyp een commando (bijv. 'pmd'): ").strip()
            if not cmd: continue
            
            msg = f"{cmd}\n"
            ser.write(msg.encode())
            print(f"> Verzonden: {cmd}")
            
            # Wacht op antwoord
            print("  Wachten op antwoord van ESP32...")
            time.sleep(0.5) 
            
            if ser.in_waiting > 0:
                response = ser.read_all().decode(errors='replace').strip()
                print(f"< ONTVANGEN: {response}")
            else:
                print("  (Geen antwoord ontvangen)")

    except KeyboardInterrupt:
        print("\nTest gestopt.")
    finally:
        if ser: ser.close()

if __name__ == "__main__":
    test_connection()