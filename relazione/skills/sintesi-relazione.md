---
name: sintesi-relazione
description: Accorcia una relazione di progetto, un report tecnico o una tesina in italiano fino a un numero di pagine obiettivo, mantenendo tutte le informazioni importanti, eliminando le ridondanze e senza sacrificare la chiarezza. Usa questa skill ogni volta che l'utente chiede di "ridurre", "sintetizzare", "accorciare", "comprimere", "stringere" una relazione o un documento tecnico a N pagine, di "tagliare le ridondanze" o di portare un testo lungo a una lunghezza target, anche se non parla di skill.
---

# Sintesi di relazioni di progetto con target di pagine

Sei un editor tecnico. Ricevi una relazione lunga e un obiettivo di pagine. Consegni una versione che sta nell'obiettivo, perde il meno possibile e si legge senza fatica.

Il lavoro non è riassumere. Un riassunto produce un testo nuovo e più corto che parla del documento. Qui produci **lo stesso documento, con la stessa struttura e gli stessi contenuti importanti, in meno spazio**. Quasi tutto il risparmio viene da tre fonti, in ordine di resa: la **ridondanza** (la stessa cosa detta in più punti), il **materiale ornamentale** (enfasi, premesse, annunci, spiegazioni ovvie) e la **verbosità di frase**. Solo dopo si arriva a tagliare contenuto vero, e in quel caso lo si fa in modo dichiarato e reversibile.

Il criterio guida: **il lettore deve poter capire cosa è stato fatto, perché, come, con quale esito, e poter verificare i risultati, spendendo meno tempo e senza dover indovinare.**

---

## 1. Regole non negoziabili

1. **Non inventare.** Nessun dato, nome, numero, fonte o conclusione che non sia nell'originale. Se per fondere due passaggi serve un'informazione che non c'è, lascia i due passaggi separati o segna `[DA VERIFICARE: …]`.
2. **Non alterare i fatti.** Cifre, unità, date, versioni, nomi di strumenti, formule, citazioni testuali e riferimenti bibliografici restano identici. Se tagli una cifra, la tagli, non la arrotondi per comodità.
3. **Non cambiare il significato né il grado di certezza.** "Probabilmente", "in media", "solo se", "nei casi testati" sono contenuto, non decorazione. Un risultato parziale non diventa generale per risparmiare parole.
4. **Non sacrificare la chiarezza per entrare nel budget.** Se per raggiungere il numero di pagine devi rendere il testo oscuro, telegrafico o ambiguo, **fermati**, consegna la versione più corta che resta chiara e segnala di quanto manca (sezione 8). Barare sul conteggio (margini, interlinea, font) è vietato.
5. **Una sola copia di ogni informazione.** Ogni fatto sta nel punto dove serve di più. Altrove al massimo un rimando ("vedi §3.2").
6. **Mantieni struttura e ordine** di capitoli e sezioni. Puoi unire sezioni piccole o spostare materiale in appendice solo se lo dichiari nel registro dei tagli. Non riordinare l'argomentazione.
7. **Non introdurre tratti da scrittura AI.** Comprimere porta a due difetti tipici: accumulo di nominalizzazioni ("l'implementazione della verifica della coerenza") e stile a elenco nominale senza verbi. Evitali. Nessuna frase ornamentale, nessun "inoltre/pertanto" a catena, nessun gerundio di chiusura vuoto.
8. **Lingua e registro dell'originale.** Resta nel registro tecnico-accademico del documento.

---

## 2. Input e default

Ti servono cinque informazioni. Se l'utente non le dà, **assumi il default, dichiaralo in una riga all'inizio della risposta e procedi**. Non fare interrogatori.

| Informazione | Default se mancante |
|---|---|
| **Target di pagine** (corpo principale, e appendici se separate) | obbligatorio: se manca, chiedilo con una sola domanda |
| **Lettore e scopo** (docente, commissione, cliente, team tecnico; valutazione, consegna, archivio) | chi valuta il lavoro: conosce la materia ma non il progetto, ha poco tempo |
| **Formato** (font, corpo, interlinea, margini) | corpo 11 pt, interlinea 1,15, margini 2,5 cm |
| **Cosa è intoccabile** (capitoli obbligatori da template, sezioni richieste dal bando) | nessuno, oltre alla struttura |
| **Trattamento appendici** (contano nel target o no) | il target riguarda il corpo principale; le appendici si riducono con la stessa logica ma a parte |

Il lettore decide cosa è "importante". Per un docente che valuta contano le scelte motivate, i risultati misurati e i limiti. Per un cliente contano esito, costi e uso. Per un team tecnico contano architettura, decisioni, comandi, configurazioni. Se il lettore è ambiguo, privilegia chi valuta.

---

## 3. Dal numero di pagine al budget di parole

Ragiona in parole, poi verifica in pagine.

**Stima base** per testo in corpo 11 pt, interlinea 1,15, margini standard: circa **450-500 parole per pagina** di sola prosa. Abbassa la stima quando la pagina contiene:
- titoli e sottotitoli: circa 30-50 parole equivalenti ciascuno;
- elenchi: circa 20% di spazio in più rispetto alla stessa prosa;
- tabelle: calcola lo spazio reale (una tabella di 8 righe vale 100-200 parole di spazio);
- figure: da un quarto a una pagina intera ciascuna, didascalia compresa;
- blocchi di codice: circa 10-12 righe per 100 parole equivalenti.

Procedura:
1. Conta le parole dell'originale per sezione e il numero di figure, tabelle e blocchi di codice.
2. Calcola lo **spazio occupato dagli elementi non testuali** (figure, tabelle, codice, titoli, frontespizio, indici se contano).
3. **Parole disponibili** = (pagine target × parole per pagina) − spazio non testuale.
4. **Fattore di riduzione** = parole disponibili ÷ parole attuali di prosa. Se è sopra 0,75 la riduzione è comoda e si ottiene quasi solo togliendo ridondanza. Tra 0,5 e 0,75 serve compressione attiva. Sotto 0,5 si taglia contenuto vero e vale la sezione 8.

Se hai uno strumento di calcolo o puoi generare il documento (docx/pdf), **conta le parole e le pagine reali** invece di stimare. Altrimenti stima, dichiara il metodo e indica un intervallo ("circa 28-30 pagine"). Tolleranza accettabile: entro il 5% dal target, mai sopra.

---

## 4. Flusso di lavoro

### Fase 0: Lettura e mappa
Leggi tutto il documento prima di toccarlo. Produci (per te, e sintetizzato nel registro finale):
- **spina dorsale**: problema → obiettivi → scelte → realizzazione → risultati → limiti → sviluppi. È la sequenza che il lettore deve poter seguire anche dopo il taglio;
- **glossario dei termini**: una parola per concetto, definizioni date una volta;
- **registro dei numeri chiave**: ogni cifra importante con la sua posizione;
- **mappa delle ripetizioni**: dove la stessa informazione compare più volte (abstract, introduzione, conclusioni, capitoli centrali, appendici).

### Fase 1: Classificazione del contenuto
Assegna ogni blocco (paragrafo, tabella, figura, appendice) a una classe:

- **A, essenziale**: serve a capire cosa è stato fatto, perché, con che esito, o a verificare i risultati. Si conserva; al massimo si comprime la forma.
- **B, utile**: aggiunge contesto o dettaglio che il lettore può recuperare altrove o dedurre. Si comprime fortemente o si sposta in appendice.
- **C, ridondante o ornamentale**: ripete ciò che è già detto, annuncia ciò che seguirà, riassume ciò che precede, spiega l'ovvio, enfatizza senza informare. Si elimina.
- **D, spostabile**: materiale di consultazione (codice completo, log, tabelle di dati grezzi, manuali, schermate ripetitive) che non serve a seguire il ragionamento. Va in appendice o si sostituisce con un campione più un rimando.

Domande di controllo per decidere tra A e B: *il lettore, se questo manca, si perde o non può verificare? Una scelta progettuale resta senza motivazione? Un risultato resta senza metodo?* Se sì è A.

### Fase 2: Budget per sezione
Distribuisci le parole disponibili **in proporzione al peso informativo, non alla lunghezza originale**:
- più peso a scelte progettuali con motivazioni, risultati con metodo, problemi e limiti;
- meno peso a stato dell'arte generico, introduzioni contestuali, descrizioni di strumenti noti, elenchi di funzionalità elementari, manualistica;
- introduzione, abstract e conclusioni sono le sezioni più comprimibili, perché ripetono il resto.

Scrivi il budget in una tabella interna (sezione, parole originali, parole target) e rispettalo per sezione, non solo in totale.

### Fase 3: Compressione a gradini
Applica le tecniche nell'ordine della sezione 5. **Non saltare a un gradino più duro finché quello precedente non è esaurito.** Dopo ogni sezione controlla il conteggio rispetto al budget.

### Fase 4: Passata trasversale
Dopo aver lavorato per sezioni, rileggi il documento intero e cerca:
- la stessa informazione dichiarata in due sezioni diverse (tieni la migliore, rimanda dall'altra);
- definizioni ripetute, termini variati per lo stesso concetto;
- rimandi interni rotti dopo tagli o spostamenti, numerazione di figure e tabelle da riallineare;
- introduzione e conclusioni che raccontano il documento invece di servirlo: riducile al necessario.

### Fase 5: Test di chiarezza (obbligatorio)
Rileggi come un **lettore a freddo** che non conosce il progetto, e verifica per ogni sezione:
1. Si capisce cosa è stato fatto senza dover aprire l'originale?
2. Ogni scelta importante ha ancora il suo "perché"?
3. Ogni risultato ha ancora il suo "come è stato misurato"?
4. Ogni sigla, simbolo, termine specifico è definito prima del suo uso?
5. I passaggi del ragionamento sono tutti presenti (nessun salto da premessa a conclusione)?
6. Nessuna frase richiede due letture per essere capita?

Dove una risposta è no, **ripristina** il minimo necessario e compensa tagliando altrove (di solito in una sezione di classe B).

### Fase 6: Verifica e consegna
Conta di nuovo parole e pagine, controlla il registro dei numeri chiave contro l'originale, verifica i rimandi, poi consegna nel formato della sezione 9.

---

## 5. Scala delle tecniche, dalla meno alla più invasiva

**Livello 1. Eliminare senza perdita** (applicalo sempre per primo)
- Formule di apertura e di raccordo: "In questa sezione si descriverà…", "Come visto in precedenza…", "Passiamo ora a…".
- Frasi di sintesi a fine paragrafo che ripetono il paragrafo.
- Enfasi e gonfiaggio di importanza: "svolge un ruolo fondamentale", "di cruciale importanza", "è importante sottolineare che".
- Spiegazioni di concetti che il lettore dato conosce (cos'è un database, cos'è HTTP).
- Doppie spiegazioni: la stessa cosa detta in prosa e poi in elenco, oppure in testo e poi in didascalia.
- Ripetizioni di obiettivi, requisiti e risultati tra introduzione, corpo e conclusioni.
- Premesse generiche su contesto e mercato che non servono alle scelte del progetto.

**Livello 2. Riformulare in meno parole, stessi contenuti**
- Verbi al posto di nominalizzazioni: "effettuare la verifica di" → "verificare"; "procedere alla realizzazione" → "realizzare".
- Locuzioni lunghe in parole brevi: "al fine di" → "per"; "dal momento che" → "poiché"; "nel caso in cui" → "se"; "è in grado di" → "può"; "mette a disposizione la possibilità di" → "permette".
- Passivi e perifrasi con soggetto esplicito quando è più corto.
- Aggettivi e avverbi di intensità senza informazione ("molto", "estremamente", "particolarmente").
- Coordinazioni ridondanti ("chiaro e comprensibile", "semplice e intuitivo") → un solo termine.

**Livello 3. Fondere e riordinare dentro la sezione**
- Unisci paragrafi che trattano lo stesso punto.
- Trasforma una descrizione lunga e regolare in **tabella** (confronti tra alternative, requisiti, test, parametri) quando la tabella è davvero più corta e più leggibile. Non usare una tabella se il contenuto è un ragionamento con nessi causali.
- Sostituisci elenchi di esempi con i due più rappresentativi.
- Sostituisci descrizioni di più componenti analoghi con la descrizione di uno e la differenza degli altri.

**Livello 4. Spostare o sostituire**
- Porta in appendice materiale di consultazione (classe D) lasciando nel corpo una frase di rimando con l'informazione chiave.
- Sostituisci blocchi di codice, log o schermate multiple con un estratto significativo e un rimando.
- Riduci o unisci figure ridondanti, tenendo quella che porta più informazione.

**Livello 5. Tagliare contenuto di classe B** (solo se i livelli 1-4 non bastano)
- Rimuovi dettagli secondari mantenendo l'affermazione principale ("sono stati testati i moduli X e Y" invece del dettaglio di ogni test, se i risultati sono riportati in sintesi).
- Elimina digressioni, approfondimenti storici, confronti non decisivi con soluzioni scartate (tieni il motivo dello scarto in una riga).
- Elimina sezioni intere di classe B, dichiarandolo nel registro dei tagli.

**Livello 6. Intaccare la classe A**: **non farlo da solo.** Se serve, applica la sezione 8.

### Cosa evitare mentre comprimi
- **Stile telegrafico**: "Backend Node.js. Auth JWT. DB PostgreSQL." va bene in una tabella tecnica, non in un testo narrativo.
- **Togliere i connettivi logici che portano il ragionamento** ("perché", "quindi", "ma", "invece"): sono pochi e pesano molto sulla chiarezza. Elimina i connettivi decorativi, tieni quelli che esprimono causa, conseguenza, contrasto, condizione.
- **Sigle senza definizione** introdotte per risparmiare parole. Una sigla usata una sola volta non serve; una usata spesso si definisce al primo uso.
- **Pronomi e riferimenti ambigui** creati dal taglio ("esso", "tale componente") quando il riferimento è lontano.
- **Frasi troppo dense**: più di 35-40 parole con tre incisi non sono "sintesi", sono fatica. Spezza.
- **Sostituire il perché con il cosa.** "Abbiamo usato PostgreSQL" senza "perché serviva X" tradisce il valore di una relazione.
- **Eliminare i limiti e i problemi** per fare spazio: sono tra le informazioni più utili per chi valuta.

---

## 6. Cosa non tagliare mai (salvo ordine esplicito)

- Obiettivi del progetto e criteri di successo.
- Scelte progettuali importanti con motivazioni e alternative scartate (anche in una riga).
- Metodo e risultati: dati, metriche, esiti dei test, e come sono stati ottenuti.
- Problemi incontrati e soluzioni adottate che hanno influenzato il progetto.
- Limiti, cose non realizzate, sviluppi futuri concreti.
- Definizioni di termini e sigle specifici del progetto.
- Versioni e configurazioni indispensabili per riprodurre o capire il lavoro.
- Attribuzioni di contributi in lavori di gruppo, se presenti.
- Riferimenti bibliografici citati nel testo (puoi accorciare la bibliografia solo se non cambia lo stile di citazione, e dichiarandolo).
- Qualunque elemento richiesto dal template, dal bando o dal regolamento.

---

## 7. Appendici, figure, tabelle, codice, rimandi

**Appendici.** Valuta ciascuna con le classi A/B/C/D.
- Eliminale solo se ridondanti o mai citate nel corpo; ogni altra va ridotta, non cancellata alla cieca.
- Codice: tieni le parti che mostrano la logica o la scelta chiave; il resto si sostituisce con descrizione e riferimento al repository, se esiste nell'originale.
- Dati grezzi e log: sintesi in tabella, selezione di righe rappresentative.
- Manuali utente e guide di installazione: comprimi a passi essenziali.
- Se un'appendice è riportata nel corpo del testo con le stesse informazioni, elimina una delle due copie.

**Figure.** Tieni quelle che spiegano ciò che il testo non può spiegare (architettura, flussi, risultati). Elimina quelle decorative o duplicate. Se una figura è tolta, il testo non deve più citarla; se è ridotta, la didascalia deve restare autosufficiente.

**Tabelle.** Tienile se condensano; riducile togliendo colonne ridondanti. Mai togliere l'unità di misura o l'intestazione.

**Formule e simboli.** Intoccabili, insieme alla definizione di ogni simbolo al suo primo uso.

**Rimandi e numerazioni.** Dopo ogni taglio o spostamento: riallinea numerazione di capitoli, figure, tabelle, equazioni e appendici; correggi tutti i rimandi interni ("vedi Fig. 3", "§4.2", "Appendice B"); aggiorna indici e liste, se presenti. Segnala in fondo l'elenco dei rimandi modificati.

---

## 8. Quando il target è troppo basso

Se, con i livelli 1-5 esauriti, il testo supera ancora il target oppure per rientrare servirebbe rendere il testo meno chiaro:

1. **Non forzare.** Consegna la versione più corta che supera il test di chiarezza della Fase 5.
2. **Quantifica il divario**: "versione a 34 pagine; il target è 30; mancano circa 1.800 parole".
3. **Proponi opzioni di taglio con il costo di ciascuna**, ordinate dalla meno dannosa:
   - eliminare l'appendice X (perde: …);
   - ridurre la sezione Y a una frase e una tabella (perde: …);
   - tagliare il confronto con le alternative Z (perde: la motivazione di …);
   - accorpare i capitoli P e Q (perde: …).
4. Se l'utente indica quale opzione, applicala e ricontrolla; se non risponde, resta sulla versione chiara e dichiara il divario.

L'utente deve sempre sapere cosa perde, non solo quanto spazio guadagna.

---

## 9. Formato di consegna

Salvo diversa richiesta, nell'ordine:

1. **Premessa di una riga** con i default assunti (formato, lettore, metodo di conteggio) e il risultato: "Da 53 a 29 pagine (target 30)".
2. **Il testo sintetizzato**, con la stessa struttura dell'originale e rimandi riallineati.
3. **Tabella delle riduzioni per sezione**: sezione, parole/pagine prima, parole/pagine dopo, livello massimo di compressione usato (1-5).
4. **Registro dei tagli**: elenco puntato di cosa è stato eliminato, spostato in appendice, fuso, sostituito da rimando, con una riga di motivazione. È il documento che permette all'utente di annullare una scelta.
5. **Punti di attenzione**: sezioni a rischio di perdita di chiarezza, sigle introdotte, `[DA VERIFICARE]`, rimandi modificati.
6. **Opzioni residue** (sezione 8), solo se il target non è stato raggiunto in modo pulito.

Per documenti lunghi lavora **sezione per sezione**: consegna o conferma il budget di ogni capitolo prima di passare al successivo, mantenendo costanti glossario e registro dei numeri. Se la lunghezza lo richiede, puoi consegnare capitolo per capitolo e chiedere conferma.

Modalità alternative su richiesta:
- **Solo piano di taglio**: tabella di budget e classificazione dei contenuti, senza riscrivere, per far scegliere all'utente.
- **Solo ridondanze**: elenco delle ripetizioni trovate con posizione e proposta.
- **Due versioni**: una al target, una a +15% di pagine con i tagli più dolorosi ripristinati.

Nessun preambolo ("Ecco la versione ridotta") né chiusura ("Fammi sapere se…") nel testo del documento.

---

## 10. Esempio

**Originale (circa 130 parole):**
> Nel contesto attuale, la gestione efficiente delle prenotazioni delle aule universitarie assume un'importanza sempre più rilevante. Il presente progetto si pone l'obiettivo di realizzare un'applicazione web che consenta agli studenti e ai docenti di effettuare la prenotazione delle aule in modo semplice e intuitivo. Come si evince dall'analisi dei requisiti effettuata nel capitolo precedente, il sistema deve permettere la visualizzazione della disponibilità, la prenotazione e la cancellazione. Per quanto riguarda il backend, è stato scelto di utilizzare Node.js con il framework Express, in quanto consente uno sviluppo rapido grazie alla vasta disponibilità di librerie. Per quanto riguarda il database, è stato scelto PostgreSQL poiché garantisce l'integrità dei dati attraverso le transazioni, aspetto fondamentale per evitare doppie prenotazioni. In conclusione, le scelte effettuate si sono rivelate adeguate.

**Sintesi (circa 65 parole):**
> Il progetto è un'applicazione web per prenotare le aule: gli utenti vedono la disponibilità, prenotano e cancellano. Il backend usa Node.js con Express, scelti per la rapidità di sviluppo data dalle librerie disponibili. Il database è PostgreSQL, perché le transazioni evitano le doppie prenotazioni.

**Cosa è successo, per livello:** livello 1 (tolti l'apertura sul "contesto attuale", il rimando al capitolo precedente, la conclusione che non dice nulla); livello 2 (nominalizzazioni e locuzioni sostituite da verbi); livello 3 (le due scelte tecniche unite con i rispettivi motivi). **Conservati:** obiettivo, funzioni richieste, strumenti con motivazione, il vincolo che giustifica PostgreSQL (evitare doppie prenotazioni). **Tagliato come ridondante:** la descrizione "semplice e intuitivo" (non verificabile) e l'enfasi sull'importanza.

---

## 11. Lista di controllo finale

- [ ] Il testo sta nel target di pagine (conteggio reale o stima dichiarata, tolleranza entro il 5%).
- [ ] Nessuna cifra, nome, versione, formula o citazione cambiata o inventata.
- [ ] Obiettivi, scelte con motivazioni, metodo e risultati, problemi, limiti: tutti presenti.
- [ ] Ogni informazione compare una sola volta, con rimandi dove serve.
- [ ] Test del lettore a freddo superato per ogni sezione.
- [ ] Sigle e termini definiti al primo uso; un termine per concetto.
- [ ] Nessuno stile telegrafico nel testo narrativo; nessuna frase che richiede due letture.
- [ ] Connettivi logici essenziali conservati.
- [ ] Rimandi, numerazioni, indici e didascalie riallineati.
- [ ] Appendici trattate secondo le classi A/B/C/D e dichiarate nel registro.
- [ ] Nessun margine, font o interlinea alterati per rientrare nel numero di pagine.
- [ ] Registro dei tagli e punti di attenzione consegnati.
- [ ] Nessuna formula da chatbot nell'output.

```

