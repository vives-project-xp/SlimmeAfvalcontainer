# Hardwaredetails LEDs en ultrasoon

Deze map bevat de detaildocumentatie die vroeger los in
`Documentaties/RaspberryPi5` stond.

## Inhoud

- [ledstrip.md](ledstrip.md): LED-strip hardware, GPIO-pinnen, kleuren,
  commando's en process isolation.
- [ultrasone.md](ultrasone.md): HC-SR04 sensoren, pinnen, vullingsgraad en
  inworp-detectie.
- [bedrading.md](bedrading.md): overzicht van de bedrading en aansluitingen.

## Koppeling met de runtime

De GUI gebruikt:

```text
led_controller.py
ultrasone_controller.py
```

In de huidige Pi-installatie moeten vooral deze twee paden gecontroleerd worden:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel/led_controller.py
/home/kobe/SlimmeAfvalcontainer/Code PI/led_controller.py
```

De tweede is belangrijk omdat de LED-subprocess-code de controller uit
`Code PI` kan importeren.

