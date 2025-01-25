# Progetto Super Mario Bros IA

## Descrizione del Progetto
Questo progetto mira a implementare **tecniche avanzate di Reinforcement Learning (RL)** per addestrare un agente in grado di **giocare e completare autonomamente** un livello del classico videogioco Super Mario Bros [1]. L'obiettivo è esplorare l'efficacia di diversi algoritmi di RL e l'integrazione con sistemi di visione artificiale per migliorare le capacità dell'agente [2].

## Obiettivi
Gli obiettivi principali del progetto includono [2, 3]:
*   Implementazione e confronto di algoritmi di Reinforcement Learning, specificamente **DDQN (Deep Double Q-Network)** e **PPO (Proximal Policy Optimization)**.
*   Integrazione di un sistema di visione artificiale basato su **YOLOv5** per fornire all'agente una comprensione più dettagliata dell'ambiente.
*   Valutazione delle prestazioni dei modelli in termini di capacità di completare il livello e tempi di addestramento.

## Metodologia
Il progetto è strutturato seguendo questi passaggi principali [4]:
*   **Background Teorico:** Introduzione ai concetti fondamentali di Reinforcement Learning, DDQN, PPO e YOLOv5 [5].
*   **Metodologia:** Descrizione dell'ambiente di lavoro (Gym e Gym-Super-Mario-Bros), creazione di un dataset personalizzato per YOLOv5, implementazione dei modelli di RL e integrazione con YOLOv5 [6, 7].
*   **Analisi dei Risultati:** Valutazione delle prestazioni dei modelli, confronto tra DDQN, PPO e PPO+YOLOv5 [8].
*   **Problemi e Soluzioni:** Discussione delle difficoltà incontrate e delle soluzioni adottate [9].
*   **Conclusioni:** Riepilogo dei risultati principali e prospettive future [10].

### Ambiente di Lavoro
*   **Gym e Gym-Super-Mario-Bros:** Utilizzo dell'ambiente di simulazione del videogioco Super Mario Bros per l'addestramento dell'agente [11].
*   **Livello:** SuperMarioBros-1-1-v0 [11].
*   **Azioni:** Sono stati utilizzati set di azioni **SIMPLE_MOVEMENT** e **COMPLEX_MOVEMENT**, senza modifiche sostanziali all'ambiente [11, 12].
*   **Wrapper:** Sono stati implementati wrapper personalizzati come **FrameStack**, **ResizeObservation**, **GrayscaleObservation** e **Reward Shaping** per ottimizzare l'apprendimento [12].

### Dataset YOLOv5
*   **Creazione:** Dataset di circa 350 immagini, raccolte da frame di gioco e dataset preesistenti [13].
*   **Annotazione:** Annotazione manuale tramite Roboflow, con 10 classi principali: castle, interactable, fm, fpole, hole, goomba, pipe, mr, sm, turtle [14, 15].
*   **Suddivisione:** 85% delle immagini per il training, 15% per la validazione [16].
*   **Addestramento:** YOLOv5l con parametri specifici (dimensione immagine 320x320, batch size 8, epoche 50) [17, 18].

### Implementazione dei Modelli
*   **DDQN:** Implementazione con PyTorch, CNN per elaborare i frame di gioco, Replay Buffer, learning rate 0.0001, gamma 0.99, replay buffer size 100.000 [19, 20].
*   **PPO:** Implementazione con Stable-Baselines3, due configurazioni principali: 512 passi con learning rate 0.0000005 e 2048 passi con learning rate 0.000005 [20, 21].
*   **PPO + YOLOv5:** Integrazione tramite un wrapper che elabora i frame con YOLOv5, aggiungendo un canale di osservazione con i bounding box [22].

## Risultati
I principali risultati ottenuti sono:
*   **DDQN:** Raggiunge una vittoria su 1000 episodi, con una convergenza lenta ma stabile [23, 24].
*   **PPO (512 passi):** Ottiene 54 vittorie su 10 milioni di passi, ma con prestazioni variabili e instabili [24, 25].
*   **PPO (2048 passi):** Non raggiunge nessuna vittoria [24, 26].
*   **PPO + YOLOv5:** Non completa alcun livello, nonostante l'elevata accuratezza di YOLOv5 nel riconoscimento degli oggetti [27, 28].
  *   Precisione del 94.2%, Recall del 100%, mAP@50-95 dell'87.6% [29].

## Problemi Incontrati
*   **Limitazioni Hardware:** Problemi con l'utilizzo della GPU su Mac, incompatibilità dei risultati YOLOv5 tra sistemi operativi [30].
*   **Difficoltà nel Salto:** Incapacità di superare il terzo tubo da parte di tutti i modelli, nonostante tentativi di reward shaping e azioni personalizzate [31, 32].
*   **Tempi di Addestramento:** Tempi lunghi per l'addestramento del modello DDQN [33].
*   **Dataset YOLOv5:** Problemi di bilanciamento delle classi nel dataset [34].

## Conclusioni
Il progetto ha dimostrato il potenziale e le sfide dell'applicazione di tecniche avanzate di Reinforcement Learning e visione artificiale nel contesto di un videogioco. Sebbene alcuni modelli abbiano mostrato la capacità di apprendere strategie vincenti, rimangono margini di miglioramento soprattutto per quanto riguarda la stabilità e la generalizzazione delle strategie [35, 36].

## Sviluppi Futuri
*   Ottimizzazione degli iperparametri per PPO [37].
*   Miglioramento del dataset YOLOv5, con un maggior numero di immagini e un bilanciamento delle classi [38].
*   Esplorazione di modelli ibridi che combinino DDQN e PPO [38].
*   Sperimentazione con tecniche di reward shaping avanzate e curiosity-driven exploration per superare ostacoli complessi [38].
*   Ottimizzazione dell'integrazione tra RL e YOLOv5 per ridurre la complessità computazionale [39].

## Strumenti e Hardware Utilizzati
*   **Librerie e Framework:** PyTorch, Stable-Baselines3, YOLOv5l, OpenCV, Gym, CUDA [40].
*   **Strumenti Dataset:** Roboflow [40].
*   **Hardware:**
    *   Macchina Windows con GPU NVIDIA GeForce RTX 4050 Laptop GPU per l'addestramento di DDQN e PPO [41].
    *   Mac Studio con Chip Apple M2 Ultra per l'addestramento del modello PPO integrato con YOLOv5 [41].

## Contatti
*   Arcangeli Giovanni
*   Ciancio Vittorio
*   Di Maio Marco
