# Analyse af benchmarking-struktur

- Et program kører 1 udregning på noget input data.
- Input data kan ligge i en fil
- Fordel i at den samme data bliver brugt hver gang.

- Programmet outputter siden kørselstid i en fil
- Output format-muligheder
  - .csv
  - .json
  - .txt
- Output entries består af
  - Program-navn (algoritme-navn)
  - Input-datastørrelse
  - Kørselstid
- Output data skal kunne læses, så entries med samme program-navn kan sættes sammen

- Diagrammet skal plotte en graf hvor hver linje har en farve som tilhører et særligt program (algoritme).
- Ud af x-aksen er inputstørrelsen
- Ud af y-aksen er kørselstiden

## Løsningsforslag

- [ChatGPT forslag](https://chat.openai.com/share/f7338b9c-553b-406f-8a60-9acebf53a9cd)
- csv-fil med data i formatet:
  - algorithm-name,input-size,run-time
- C-program med main metode, der kører 1 algortime og logger hvor lang tid kørselstiden er, og inputstørrelsen
- C-program tager 3 argumenter: navn på metode, input-størrelse, navn på csv-fil den skal gemme i
- Python læser csv-filen og opretter et dictionary som indeholder alt data.

Eksempel-csv:

    CPU-matrix-sum, 10,     0.1
    CPU-matrix-sum, 100,    1.2
    CPU-matrix-sum, 1000,   9.8
    CPU-matrix-sum, 10000,  101.2
    GPU-matrix-sum, 10,     0.1
    GPU-matrix-sum, 100,    1.2
    GPU-matrix-sum, 1000,   9.8
    GPU-matrix-sum, 10000,  101.2

Eksempel output-datastructure:  
Hver (algorithm,inputsize) skal mappe til en tid

    input-sizes = [10, 100, 1000, 10000]

    dictionary = { 
        "CPU-matrix-sum": { 
            10: 0.1,
            100: 1.2,
            1000: 9.8,
            10000: 101.2,
        },
        "GPU-matrix-sum": { 
            10: 0.1,
            100: 1.2,
            1000: 9.8,
            10000: 101.2,
        },
    }

Derefter noget kode som tager dictionariet og bruger det med matplotlib.

## Fixes

- Verify at CLOCKS_PER_SEC er korrekt.
