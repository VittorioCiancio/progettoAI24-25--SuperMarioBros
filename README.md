# Progetto Super Mario Bros: Reinforcement Learning e Visione Artificiale
![Anteprima del Progetto](assets/demo.gif)
Questo progetto nasce come elaborato per l'esame di Intelligenza Artificiale e si propone di esplorare le applicazioni del **Reinforcement Learning (RL)** in un ambiente dinamico e complesso: il videogioco **Super Mario Bros**. L'obiettivo principale è stato quello di sviluppare un agente autonomo in grado di apprendere e completare un livello di gioco attraverso tecniche avanzate di RL, integrando anche la visione artificiale per potenziare le capacità decisionali. Questo lavoro mira a valutare l'efficacia di diversi approcci, mettendo in luce sia i successi che le limitazioni incontrate.

## Contesto e Obiettivi del Progetto

Il progetto si inserisce nel contesto dell'esame di Intelligenza Artificiale, con lo scopo di applicare tecniche avanzate di RL in un ambiente sfidante come Super Mario Bros. L'obiettivo principale era di sviluppare un agente capace di completare un livello del gioco in modo autonomo, ottimizzando le sue prestazioni attraverso l'addestramento con tecniche di RL avanzate. Nello specifico, ci siamo posti i seguenti sotto-obiettivi:

*   **Implementazione e confronto di algoritmi di RL**: Abbiamo esaminato e confrontato due approcci principali, **DDQN (Deep Double Q-Network)** e **PPO (Proximal Policy Optimization)**, per valutarne i punti di forza e le debolezze in un contesto pratico.
*   **Integrazione di un sistema di visione artificiale**: Abbiamo integrato **YOLOv5**, un framework per il riconoscimento visivo, per fornire all'agente informazioni dettagliate sull'ambiente e migliorare la qualità delle sue decisioni.

L'intento non era solo quello di massimizzare le prestazioni dell'agente, ma anche di esplorare le sinergie tra RL e visione artificiale, aprendo la strada a nuove applicazioni in scenari complessi.

## Metodologia

Il progetto è stato strutturato seguendo un approccio metodologico ben definito:

*   **Ambiente di lavoro**: Abbiamo utilizzato il framework **Gym** con il pacchetto **Gym-Super-Mario-Bros**, in particolare il livello iniziale SuperMarioBros-1-1-v0, senza apportare modifiche sostanziali all'ambiente. Abbiamo, però, introdotto wrapper personalizzati come **FrameStack**, **ResizeObservation** e **GrayscaleObservation** per semplificare l’elaborazione dei frame e fornire contesto temporale. È stato anche implementato un **reward shaping** per incentivare l’agente a progredire nel livello.

*   **Dataset YOLOv5**: Per l'addestramento di YOLOv5, abbiamo creato un **dataset personalizzato di circa 350 immagini**, annotate con lo strumento **Roboflow**, e suddiviso in training (85%) e validazione (15%). Le 10 classi di oggetti annotate includono: castle, interactable, fm, fpole, hole, goomba, pipe, mr, sm, turtle. L'addestramento del modello YOLOv5 è avvenuto su una macchina Windows con GPU, con parametri quali dimensione immagine 320x320, batch size 8 e 50 epoche.

*   **Implementazione dei modelli**: I modelli DDQN e PPO sono stati implementati utilizzando **PyTorch** e **Stable-Baselines3**. Il modello DDQN si basa su una CNN per l'elaborazione dei frame e un replay buffer per migliorare l'efficienza dell'apprendimento.  Per il PPO, sono state utilizzate due configurazioni diverse: 512 passi con learning rate di 0.0000005, e 2048 passi con learning rate 0.000005. L'integrazione di YOLOv5 con PPO ha richiesto la creazione di un wrapper personalizzato per fornire all'agente un canale aggiuntivo con i bounding box degli oggetti rilevati.

## Risultati e Analisi

### Prestazioni dei modelli RL

L'addestramento del modello **DDQN** ha evidenziato una convergenza graduale, stabilizzando la lunghezza degli episodi intorno ai 500 passi. La ricompensa cumulativa è cresciuta progressivamente, e il modello è riuscito a completare il livello una volta su 1000 episodi, anche se in modo apparentemente casuale.

Il modello **PPO** è stato addestrato con due configurazioni:
*   Con **512 passi** per aggiornamento, il modello ha mostrato un andamento non lineare nelle vittorie, totalizzando 54 successi su 10 milioni di passi, ma con prestazioni altalenanti.
*   Con **2048 passi** per aggiornamento, il modello non ha ottenuto alcuna vittoria durante i 10 milioni di passi, evidenziando difficoltà nell'apprendimento di strategie efficaci.

### Integrazione con YOLOv5

L'integrazione di **PPO e YOLOv5** ha portato a risultati interessanti. YOLOv5 ha mostrato ottime performance nel riconoscimento degli oggetti, con una precisione del 94.2% e un recall del 100%. Tuttavia, l'agente addestrato con PPO e YOLOv5 non è riuscito a completare alcun livello, dimostrando difficoltà nel tradurre le informazioni visive in azioni vincenti.

### Confronto tra i modelli

In sintesi, i risultati ottenuti mostrano:

*   Il modello **DDQN** ha mostrato una maggiore stabilità e robustezza, completando il livello una volta, anche se con tempi di addestramento lunghi.
*   Il modello **PPO** con 512 passi ha ottenuto più vittorie (54), ma con prestazioni instabili. La configurazione con 2048 passi non ha prodotto risultati positivi.
*   Il modello **PPO + YOLOv5**, nonostante le ottime performance di YOLOv5 nel riconoscimento degli oggetti, non è riuscito a completare il livello.

## Sfide e Ostacoli Incontrati

Durante il progetto, abbiamo incontrato diverse sfide:

*   **Limitazioni hardware**: L'utilizzo limitato della GPU su Mac ha rallentato l'addestramento del modello PPO + YOLOv5. L'incompatibilità dei risultati di YOLOv5 tra Windows e Mac ha comportato la necessità di ripetere l'addestramento su Mac.
*   **Difficoltà nel superare ostacoli specifici**: Nessun modello è stato in grado di superare in modo affidabile il salto del terzo tubo, il più alto del livello.
*   **Tempi di addestramento**: Il modello DDQN ha richiesto tempi di addestramento molto lunghi per raggiungere la convergenza.
*   **Bilanciamento del dataset YOLOv5**: Il dataset presentava un bilanciamento non ottimale delle classi, con un numero limitato di esempi per alcune categorie.

## Conclusioni e Prospettive Future

Nonostante le difficoltà, il progetto ha dimostrato il potenziale del Reinforcement Learning e della visione artificiale applicata ai videogiochi, aprendo la strada a possibili sviluppi futuri. Le principali direzioni per la ricerca futura includono:

*   **Ottimizzazione degli iperparametri del PPO**: Analizzare in dettaglio le configurazioni del PPO (passi per aggiornamento, learning rate) per migliorare la stabilità e le prestazioni.
*   **Miglioramento del dataset di YOLOv5**: Espandere il dataset e bilanciare le classi per un riconoscimento più accurato degli oggetti.
*   **Esplorazione di modelli ibridi**: Combinare le capacità esplorative del PPO con la robustezza del DDQN.
*   **Gestione di azioni complesse**: Sperimentare con reward shaping più mirati e tecniche di esplorazione avanzate per superare ostacoli critici.
*   **Riduzione della complessità computazionale**: Ottimizzare l'integrazione tra RL e YOLOv5 per ridurre i tempi di addestramento senza compromettere le prestazioni.

Questo progetto rappresenta un punto di partenza per ulteriori esplorazioni nel campo dell'intelligenza artificiale applicata al gaming, evidenziando sia le opportunità che le sfide da affrontare.

## Strumenti Utilizzati

*   **Librerie e Framework**: PyTorch, Stable-Baselines3, YOLOv5l, OpenCV, Gym, CUDA.
*   **Strumenti per il dataset**: Roboflow per l'annotazione automatica.
*   **Hardware**:
    *   Macchina Windows con GPU NVIDIA GeForce RTX 4050 Laptop GPU per l'addestramento di DDQN e PPO.
    *   Mac Studio con Chip Apple M2 Ultra per l'addestramento del modello PPO integrato con YOLOv5.


## Accesso al Progetto Completo
Per visualizzare il progetto completo (codice, dataset e risultati) è possibile accedere alla cartella condivisa su Google Drive:

[**Progetto SuperMario**](https://drive.google.com/drive/u/2/folders/1JG1K9eyDDPCn6m_y4dmqNnEn63csLQ-9)

