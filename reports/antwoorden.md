# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1
---
### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?
- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?


 ## <span style="color:Green">Antwoord 1a</span>

Ik vind de architectuur die mijn collega gekozen heeft een begrijpelijke keuze. Aangezien  het een eenvoudige architectuur is en relatief makkelijk toe te passen model is. Wel vraag ik mij af of dit de juiste keuze is voor dit specifieke probleem.

Sterke kanten: 
- De eenvoud helpt om overfitting te voorkomen.
- Het model is snel en is ideaal als 'baselinemodel'. 

Zwakke kanten: 
- Sluit niet 'goed' aan op dataset met met groot aantal features en meer dan twee dimensies.

Voor de aangeleverde dataset en dit specifieke probleem levert het gekozen linear model waarschijnlijk problemen op. Dit model kan moeilijk omgaan met drie dimensies. De dataset bestaat uit drie dimensies (timeseries). Mijn advies zou zijn om een RNN model te gebruiken. Een RNN model is geschikt voor drie dimensies.

De keuzes van de LinearConfig:
- Input=13, --> Goede keuze. Is gelijk aan het aantal attributen. 
- Output=20, --> Goede keuze. Is gelijk aan het aantal classes. 
- H1=100, --> Goede keuze. Persoonlijk kies ik voor het aantal 128 of 64 puur uit gewenning en aangeleerd in de cursus Machine Learning. Daarnast zal ik gaan experimenteren met het aantal units. Bijvoorbeeld door te verhogen naar 256 of verlagen naar 64.
- H2=10, --> Matige keuze. Te klein vergeleken met H1.
- Dropout=0.5 Matige keuze. Hierbij valt de helft af. De data is al niet groot (8800). Ik zal kiezen voor een dropout van 0.2. Op basis van de grootte van de dataset. 
---

---
## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
- Hoe had hij dit ook kunnen oplossen?
- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

 ## <span style="color:Green">Antwoord 1b</span>
 Het effect is dat het aantal dimensies wordt teruggebracht. Dit door het gemiddelde te nemen van de middelste dimensie. (De dim-parameter bepaalt over welke dimensie de bewerkingen worden uitgevoerd.) 

 Het probleem dat hij hiermee wil oplossen is het aansluiten van de data op het model. Het model gebruikt door mijn collega sluit aan op data met twee dimensies. Terwijl de aangeleverde data bestaat uit drie diemensies
  
  Het bijkomend probleem dat mijn collega hiermee oplost is ook wel bekend onder het fenomeen: "The curse of dimensionality”. In het kort: een dataset met uitgebreide features maakt het voorspellen door het model lastig, waardoor de performance en nauwkeurigheid in gevaar komen. 

- Een andere oplossing zou kunnen zijn om flatten toe te passen.

### Voordeel mean
Reductie in features naar 1. Perfomance gaat omhoog.

### Nadeel mean
Features worden samengevat naar een gemiddelde. Dit leidt tot minder accurate input.

### Voordeel flatten
Alle informatie wordt vastgehouden. Er verdwijnen geen features. 
### Nadeel flatten
Meer features kan leiden tot performance uitdagingen.


---
---
### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.

- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).


## <span style="color:Green">Antwoord 1c</span>
Het betreft een datasetset met time series, drie dimensies en een classificatieprobleem. Als we het schema en recap uit de cursus ML volgen dan komen we uit op RNN (recurrent neural networks) architecturen. Hierbij hebben we de keuze tussen:
Simple RNN, LSTM en GRU.
De volgende architectuur zal ik overwegen:
Starten met een inputlayer dan GRU en uiteindelijk eindigen met een linear layer.
Of starten met inputlayer dan RNN en eindigen met linear layer.

Hierbij gebruiken we de volgende layers: input - GRU - output. Of input-RNN-output.

De volgende indicatie en motivatie voor het aantal units/filters/kernelsize:

- Input: 13 -> Gelijk gezet aan het aantal attributen
- Hidden size: 64 --> Niet te *groot* bij hogere hidden size kan de performance onder druk komen door de benodigheid van meer rekenkracht.
- Output: 20 --> Gelijk gezet aan het aantal classes.
- Loss funtie: Cross entropy loss --> past bij het probleem,
- Optimizer: Adam --> bewezen als een van de beste optimizer met lage geheugenvereisten. En wordt in het algemeen gezien als de default optimizer. Tevens heb ik hiermee goede resultaten behaald bij eerdere opdrachten (tussentijdse_opdracht).
- Aantal layers: 3 of 4 - Genoeg lagen om mee te beginnen en het model te trainen.
3 naar 2 dimensies?: Door middel van 'flatten'

Ik verwacht onderstaand architectuur als meest veelbelovende:
Een RNN en dan wel de GRU variant. GRU gebruikt minder trainingsparameters en gebruikt zoals gezegd minder geheugen en voert sneller uit dan LSTM, terwijl LSTM nauwkeuriger is op een grotere dataset. De dataset is niet groot en simpel, waardoor LSTM niet nodig is.




---

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

## <span style="color:Green">Antwoord 1d</span>
Model.py en settings.py aangepast om het GRU model te laten werken. Daarnaast gekozen om een nieuw script (01_model_GRU_design) te maken. Hierdoor kan de junior collega zijn eigen script nog teruglezen ter lering en vermaak. Verder de *Makefile* aangepast om het model te kunnen runnen met bestaande commando's.

Wat opvalt is dat bij de 3e run al een accuracy van 96% wordt gehaald.Dit model bevat een hidden size van 256, 4 layers en een dropout van 0.2. Dit model heb ik uiteindelijk nogmaals gerund en daarbij werd een accuracy van **97%** gehaald. 

Verder ben ik een klein beetje doorgeslagen met het aantal runs. Dit is vooral een leerpunt voormijzelf. Niet te lang handmatig tunen. 

De volgende resultaten zijn het opvallendste en per aantal layers gesorteerd.

4 Layers:
- Accuraatheid 0,94. input=13, output=20,
hidden_size=64, num_layers=4, dropout=0.2
- Accuraatheid 0,944. input=13, output=20,
hidden_size=128, num_layers=4, dropout=0.2
- Accuraatheid 0,970. input=13, output=20,
hidden_size=256, num_layers=4, dropout=0.2

3 Layers:
- Accuraatheid 0,96. input=13, output=20, 
hidden_size=256, num_layers=3, dropout=0.2

2 Layers: 
- Accuraatheid 0,95. input=13, output=20, 
hidden_size=128, num_layers=2, dropout=0.2

In het Tensorboard overzichtplaatje hieronder valt te zien dat de runs bestaan uit 50 epochs. We zie rond 20 a 30 epochs verzadiging ontstaan. De loss buigt hierbij omhoog en de accuracy neemt niet meer toe of daalt. Ook valt op dat de learningrate in sommige gevallen stijl daalt. Dit komt door de instelling patience die op 10 epochs is gezet door de junior collega. Na 10 epoch zonder leren wordt de learningrate gewijzigd.




## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.

## <span style="color:Green">Antwoord 2a</span>
02_tune.py en settings.py (onder scripts) aangepast voor het GRU model.

Settings:<br>
Class GRUmodelConfig(BaseSearchSpace):<br>
    hidden_size: int <br>
    num_layers: int <br>
    dropout: float <br>

Class GRUmodelSearchSpace(BaseSearchSpace):<br>
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(128, 256)<br>
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 6)<br>
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.5)<br>
    batchsize: Union[int, SAMPLE_INT] = tune.randint(32, 256)<br>
    

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

Run 2:<b>
class GRUmodelSearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(128, 256)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 4)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.1, 0.3)
    batchsize: Union[int, SAMPLE_INT] = tune.randint(50, 200)

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
