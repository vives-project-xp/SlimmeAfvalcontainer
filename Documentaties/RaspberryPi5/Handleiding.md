# Raspberry Pi 5 Deploy Gids (2-Stage Model)

Deze gids start **na training**.  
Training is al klaar; we focussen alleen op wat nodig is om inference op de Pi 5 te laten werken.

modellen staan in: 
/home/kobe/SlimmeAfvalcontainer/Code PI/AI




`/home/pi/smart_bin/models/two_stage/`

Benodigd op de Pi:

- `stage1_main.onnx`
- `stage1_main.onnx.data`
- `stage2_overige.onnx`
- `stage2_overige.onnx.data`
- `two_stage_metadata.json`

## 3. Installeer runtime dependencies op de Pi 5

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime pillow numpy
```

## 4. Inference-logica op de Pi (belangrijk)

Gebruik altijd deze volgorde:

1. Draai **Stage 1 ONNX** (`stage1_main.onnx`) op de input-foto
2. Neem de hoofdklasse met hoogste score
3. Als hoofdklasse **niet** `Overige` is: dat is je eindresultaat
4. Als hoofdklasse **wel** `Overige` is: draai **Stage 2 ONNX** (`stage2_overige.onnx`)
5. Combineer resultaat als `Overige/<subklasse>`
6. Gebruik confidence-thresholds voor fallback:
   - lage stage-1 confidence -> bv fallback `Restafval`
   - lage stage-2 confidence -> fallback `Overige`

## 5. Labels niet hardcoden

Lees labels uit `two_stage_metadata.json`:

- `stage1_classes`
- `stage2_overige_classes`
- `main_label_for_stage2`
- `default_fallback`

Zo blijft de Pi-app correct als je later nieuwe modellen exporteert.

## 6. Snelle test op de Pi

Test met een paar bekende afbeeldingen uit verschillende klassen:

- Organisch
- PMD
- Papier
- Restafval
- Overige/* (minstens 2 subklassen)

Controleer of:

- Stage 1 en Stage 2 beide geladen worden
- `Overige` correct naar subklasse gaat
- fallback werkt bij onzekere beelden

## 7. Productie tip

Versiebeheer je modelmap op de Pi, bv:

- `models/two_stage/v1/...`
- `models/two_stage/v2/...`

Dan kan je veilig terugrollen als een nieuwe export slechter presteert.
