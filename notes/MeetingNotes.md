# Vejledningssamtalenoter med Peter Sestoft

## Møde 29/1 
Mødet handler mest om hvad og hvordan der skal stå i preliminary problem statement. 

”Lære at automatisere eksperimenterne med (python / bash scripts ...) Frem for at klistre ting ind i excel... Automatiske grafer. 

Opstil det som et læringsmål.  
Også vigtigt for at få forståelige/retsvisende resultater. 

Linjen for O(n^3) på en logaritmiske akse skulle gerne være en lige linje. 

Brug forskellige input størrelser til hver algoritme. 

Brug evt. (gammeldags) GNU-plot eller nyere. Tænk på det tidligt i projektet. 

Kan være en god idé at lave vores eget fra bunden så den kun kan det den skal kunne. Evt. brug python. 


## Møde før jul
Er nødt til at have et afsnit om hardware 

Nvcc (Nvidia cuda c compiler)
Test ikke oversættelse af byte code

Det giver mening at arbejde med floats og doubles

Addition kan ikke betale sig grundet memory flytning 
Multiplikation kan betale sig da det ellers er et tripple for loop. Her vil vi se en speed up. 
Inverse er mere spændende

Start med cpu c implementation 
Så en sekventiel gpu i cuda c (man bruger kun 1 core) (så forveksler man ikke cpu pointer med gpu pointer) gpu har kun 1-dim matrix, så man skal selv udregne index.
Så paralleliser gpu implementation

Evt. sammenligne forskellige GPUer

Projektbeskrivelse: Det her vil vi gøre, det her vil vi måske gøre. Kig på QR, da marcus har kigget på LU-D. Ikke kig på GJE (gauss elimination). Der er forskel på algoritmerne fordi vi arbejder med ”computertal”. 

Vi mødes ca. en gang om ugen. (Vi har kun ca 14 uger)

Læs op på: Lineær algebra, GPU hardware termer og teori, gamle specialer
