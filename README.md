# Machine-Learning-Project

# Teorie CNN — Ghid de concepte

Acest document explică **teoria** din spatele componentelor folosite frecvent în rețele neuronale convoluționale (CNN), independent de orice implementare anume. Scopul e să înțelegi *de ce* există fiecare componentă și *ce problemă rezolvă*, nu doar sintaxa unui anumit framework.

---

## 1. Convoluția (`Conv2d`)

O convoluție trece un **kernel** (o matrice mică de greutăți, ex. 3x3) peste imagine și calculează, la fiecare poziție, o sumă ponderată a pixelilor din acea zonă. Rezultatul e o **hartă de trăsături** (feature map).

De ce nu folosim direct un strat fully-connected pe imagine?

- **Localitate**: trăsăturile vizuale (margini, colțuri, textură) sunt locale. Nu are sens să conectezi fiecare pixel cu fiecare neuron.
- **Partajarea greutăților**: același kernel e aplicat peste toată imaginea. Dacă rețeaua învață să detecteze o margine verticală, o poate detecta oriunde în imagine, nu doar într-un colț specific. Asta reduce drastic numărul de parametri față de un strat dens.
- **Echivarianță la translație**: dacă obiectul se mută în imagine, răspunsul convoluției se mută la fel, nu dispare.

### `kernel_size`

Dimensiunea ferestrei care "alunecă" peste imagine (ex. 3x3, 5x5, sau asimetric ca 3x7).

- Kernel mic (3x3): captează detalii fine, are puțini parametri, e standard în arhitecturile moderne (ideea din VGG: două convoluții 3x3 la rând au aceeași "rază vizuală" ca un 5x5, dar cu mai puțini parametri și o non-liniaritate în plus).
- Kernel mare: captează context mai larg dintr-o singură mișcare, dar costă mai mulți parametri și calcul.
- Kernel asimetric (ex. lat pe o axă, îngust pe alta): util când datele au o structură direcțională — de exemplu, dacă pe o axă informația relevantă se întinde pe o plajă mai mare decât pe cealaltă, un kernel non-pătrat poate capta mai eficient acel tip de pattern.

### `stride`

Cu cât "sare" kernelul la fiecare pas.

- `stride=1`: kernelul se mută cu un pixel, harta de ieșire are aproape aceeași dimensiune ca intrarea (înainte de padding).
- `stride=2`: kernelul sare 2 pixeli, harta de ieșire se înjumătățește pe acea axă. E o formă de **subeșantionare** făcută direct în convoluție, alternativă la pooling.

### `padding`

Adăugarea de pixeli (de obicei zero) pe marginea imaginii înainte de convoluție.

De ce e nevoie de el:

- Fără padding, fiecare convoluție "mănâncă" din margini, iar imaginea se micșorează la fiecare strat. După multe straturi, ai putea rămâne fără spațiu.
- Padding-ul "same" (de obicei `padding = (kernel_size - 1) / 2` pentru stride=1) păstrează dimensiunea spațială constantă, ceea ce face mai ușor de proiectat arhitecturi adânci și de combinat hărți de trăsături de dimensiuni egale (de exemplu în conexiuni reziduale).
- Fără padding, pixelii de pe margine sunt "văzuți" de kernel mult mai rar decât cei din centru → informația de pe margini e subreprezentată. Padding-ul atenuează acest efect.

### `bias`

Fiecare filtru convoluțional poate avea și un termen de bias (constantă adunată după sumă ponderată). Când urmează imediat un strat de **Batch Normalization**, bias-ul devine redundant — BN are propriul termen de deplasare (shift) — deci se dezactivează de obicei (`bias=False`) ca să nu existe parametri inutili.

---

## 2. Batch Normalization (`BatchNorm2d`)

Normalizează ieșirile unui strat (le aduce la medie ≈0, deviație standard ≈1) folosind statistici calculate pe fiecare mini-batch în timpul antrenării, apoi aplică o scalare și o deplasare învățabile.

De ce ajută:

- **Stabilizează antrenarea**: fără normalizare, distribuția activărilor se poate schimba haotic de la un strat la altul pe măsură ce greutățile se actualizează ("internal covariate shift"), ceea ce face optimizarea instabilă.
- **Permite learning rate mai mare**: gradientii rămân într-un interval mai previzibil.
- **Efect de regularizare ușoară**: pentru că statisticile sunt calculate pe batch (deci au puțin zgomot), BN introduce o mică variație care poate reduce overfitting-ul.
- La inferență (evaluare), BN nu mai folosește statisticile batch-ului curent, ci o medie mobilă acumulată în timpul antrenării — de aceea contează diferența dintre modul `train()` și `eval()` al modelului.

---

## 3. Funcții de activare (ReLU / Leaky ReLU)

Activările introduc **non-liniaritate**. Fără ele, oricâte straturi convoluționale ai stivui, compunerea lor tot o operație liniară ar rămâne (echivalentă matematic cu un singur strat liniar), deci rețeaua nu ar putea învăța relații complexe.

- **ReLU**: `f(x) = max(0, x)`. Simplă, rapidă, dar are problema de "neuroni morți" — dacă un neuron ajunge mereu cu intrare negativă, gradientul lui devine 0 și nu mai învață niciodată.
- **Leaky ReLU**: `f(x) = x` dacă `x > 0`, altfel `f(x) = α·x` (cu α mic, ex. 0.1). Permite un gradient mic și pentru valori negative, reducând riscul de neuroni morți, păstrând în același timp simplitatea și viteza ReLU.

---

## 4. Pooling (`MaxPool2d`, subeșantionare)

Pooling-ul reduce dimensiunea spațială a hărții de trăsături prin agregarea unei zone (ex. 2x2) într-o singură valoare.

### De ce facem pooling deloc

- **Reducere de calcul**: mai puține valori de procesat în straturile următoare.
- **Creșterea câmpului receptiv**: fiecare valoare din harta redusă "vede" o zonă mai mare din imaginea originală, ceea ce ajută rețeaua să capteze structuri mai mari și mai abstracte pe măsură ce avansează în adâncime.
- **Invarianță locală mică la translație**: o deplasare de 1-2 pixeli a unui pattern în imagine tinde să dea același rezultat după max-pooling, pentru că alegem oricum valoarea maximă dintr-o vecinătate.

### `MaxPool2d` vs alte tipuri

- **Max pooling**: păstrează valoarea maximă din fereastră — ideea fiind că cea mai puternică activare dintr-o zonă e cea mai relevantă pentru prezența unei trăsături.
- **Average pooling**: păstrează media zonei — păstrează mai multă informație de context, dar diluează semnalele puternice locale.

### `kernel_size` și `stride` la pooling

Aceleași concepte ca la convoluție: `kernel_size` definește fereastra agregată, `stride` cât sare fereastra. De obicei `stride = kernel_size`, astfel încât ferestrele nu se suprapun și fiecare regiune e rezumată o singură dată.

### Pooling asimetric (ex. comprimă doar o axă, nu ambele)

Nu există nicio regulă care obligă reducerea simultană a ambelor dimensiuni spațiale. Dacă o axă conține informație mai "densă" sau mai puțin relevantă structural decât cealaltă, poți alege să comprimi agresiv doar acea axă și să păstrezi rezoluția pe cealaltă mai mult timp — alegerea depinde de natura datelor și de ce structură vrei să păstrezi cât mai mult posibil înainte de a o pierde ireversibil.

---

## 5. Conexiuni reziduale (Residual / Skip Connections)

Idee introdusă de arhitectura ResNet: în loc ca un bloc de straturi să învețe direct funcția dorită `H(x)`, blocul învață **reziduul** `F(x) = H(x) - x`, iar ieșirea finală e `F(x) + x` (intrarea adunată înapoi peste rezultat).

### De ce ajută

- **Rezolvă degradarea gradientului în rețele adânci**: pe măsură ce rețelele devin foarte adânci, gradientul calculat prin backpropagation poate deveni extrem de mic ("vanishing gradient") până ajunge la primele straturi, ceea ce blochează învățarea. Conexiunea reziduală oferă o "autostradă" prin care gradientul poate trece direct, nealterat, către straturile anterioare.
- **Mai ușor de optimizat**: e mai simplu pentru o rețea să învețe "nu schimba nimic" (adică `F(x) ≈ 0`) decât să reconstruiască funcția identitate de la zero printr-o secvență de convoluții. Asta înseamnă că adăugarea de straturi suplimentare nu ar trebui, în principiu, să înrăutățească performanța.
- **Permite rețele mult mai adânci** decât erau practice anterior, fără degradarea bruscă a acurateței care apărea la arhitecturile pur secvențiale foarte adânci.

### Shortcut/proiecție

Dacă intrarea `x` și ieșirea blocului `F(x)` au dimensiuni diferite (alt număr de canale sau altă rezoluție spațială, de obicei din cauza unui `stride > 1`), adunarea directă nu e posibilă matematic. Soluția e o **proiecție** — de obicei o convoluție 1x1 (eventual urmată de normalizare) — care doar ajustează dimensiunile intrării `x` ca să se potrivească cu `F(x)`, fără să adauge câmp receptiv suplimentar (de aceea kernel 1x1).

---

## 6. Attention / Squeeze-and-Excitation (concept general)

Ideea de bază a mecanismelor de attention în CNN-uri: nu toate canalele (sau toate regiunile) dintr-o hartă de trăsături sunt la fel de relevante pentru sarcina curentă. Un modul de attention învață să **recalibreze** importanța relativă a canalelor/regiunilor, în loc să le trateze pe toate egal.

Mecanism general (Squeeze-and-Excitation):

1. **Squeeze**: comprimă fiecare canal al hărții de trăsături într-un singur număr (de obicei prin global average pooling), obținând un rezumat global al "cât de activ" e fiecare canal.
2. **Excitation**: trece acel vector de rezumate printr-un mic mecanism dens (de obicei două straturi fully-connected cu o non-liniaritate între ele) care produce o pondere între 0 și 1 pentru fiecare canal.
3. **Recalibrare**: înmulțește harta originală de trăsături cu aceste ponderi, canal cu canal — canalele considerate mai informative sunt amplificate, cele mai puțin relevante sunt atenuate.

Beneficiul: rețeaua capătă o formă de "auto-reglare" — învață singură ce trăsături contează mai mult pentru exemplul curent, cu un cost computațional relativ mic comparat cu câștigul de performanță.

---

## 7. Global Average Pooling / `AdaptiveAvgPool2d`

Spre finalul unei rețele convoluționale, ai de obicei o hartă de trăsături de forma `[canale, înălțime, lățime]` și vrei să ajungi la un vector de forma `[canale]` pentru a-l trimite către un clasificator.

### De ce nu doar "flatten" direct

Aplatizarea directă (`flatten`) a unei hărți spațiale ar produce un vector a cărui dimensiune depinde de înălțimea și lățimea hărții — ceea ce înseamnă că rețeaua ar accepta doar imagini de o dimensiune fixă de intrare, și ar avea un strat dens enorm (un parametru per pixel per canal).

### Ce face Global Average Pooling

Calculează media tuturor valorilor dintr-o hartă de trăsături, pentru fiecare canal separat, reducând `[canale, H, W]` la `[canale, 1, 1]`.

Avantaje:

- **Număr de parametri drastic mai mic** decât un strat dens echivalent — reduce riscul de overfitting.
- **Funcționează cu orice dimensiune de intrare**: pentru că face media, nu contează cât de mare e harta spațială, rezultatul are mereu aceeași formă.
- **Interpretabilitate**: fiecare canal final poate fi văzut ca un "scor" pentru o anumită trăsătură abstractă învățată de rețea, agregat pe toată imaginea.

### `Adaptive` în `AdaptiveAvgPool2d`

Varianta "adaptive" nu cere să specifici manual `kernel_size`/`stride` — specifici doar dimensiunea de ieșire dorită (ex. 1x1), iar operația calculează automat fereastra necesară pentru a ajunge acolo, indiferent de dimensiunea de intrare. E utilă tocmai pentru a face rețeaua independentă de rezoluția exactă a imaginii de intrare.

---

## 8. Dropout

În timpul antrenării, dezactivează aleator (pune pe 0) un procent din neuroni la fiecare pas, de obicei într-un strat dens aproape de finalul rețelei.

De ce ajută:

- **Previne co-adaptarea excesivă** între neuroni — fără dropout, neuronii pot "învăța" să compenseze unii pentru alții în moduri foarte specifice setului de antrenare, ceea ce duce la overfitting.
- Poate fi văzut ca o formă de **ensemble implicit**: la fiecare pas se antrenează, practic, o sub-rețea diferită (cu alt subset de neuroni activi), iar la inferență (când dropout-ul e dezactivat) se folosește, aproximativ, media tuturor acestor sub-rețele.
- Procentul (ex. 0.4 = 40%) e un hiperparametru: prea mic nu regularizează suficient, prea mare poate încetini sau strica antrenarea pentru că rețeaua pierde prea multă capacitate utilă la fiecare pas.

---

## 9. Funcția de pierdere — Cross-Entropy și Label Smoothing

### Cross-Entropy Loss

Standardul pentru clasificare multi-clasă. Măsoară distanța dintre distribuția de probabilitate prezisă de model și distribuția "adevărată" (de obicei un vector one-hot, unde clasa corectă are probabilitate 1, restul 0). Penalizează puternic predicțiile încrezute dar greșite.

### Label Smoothing

În loc să ceri modelului să prezică probabilitate 1.0 exact pentru clasa corectă și 0.0 pentru rest, "înmoi" puțin țintele (ex. 0.9 pentru clasa corectă, restul distribuit pe celelalte clase).

De ce ajută:

- Reduce **supra-încrederea** modelului — un model antrenat fără label smoothing poate învăța să producă scoruri extreme (logits foarte mari) doar ca să se apropie cât mai mult de 1.0/0.0, ceea ce nu ajută generalizarea și poate strica calibrarea probabilităților.
- Acționează ca o formă ușoară de regularizare, similar ca efect (deși mecanism diferit) cu alte tehnici anti-overfitting.

---

## 10. Optimizator — Adam și Weight Decay

### Adam

Un algoritm de optimizare bazat pe gradient descendent, care adaptează automat rata de învățare pentru fiecare parametru individual, folosind estimări ale momentului întâi (media gradientului) și al doilea moment (varianța gradientului). În practică, converge rapid și e robust la alegerea hiperparametrilor, motiv pentru care e o alegere implicită des întâlnită.

### Weight Decay (regularizare L2)

Adaugă o penalizare proporțională cu mărimea greutăților la funcția de pierdere, încurajând modelul să prefere greutăți mai mici.

De ce ajută: greutăți foarte mari sunt adesea semn că modelul s-a "specializat" excesiv pe particularitățile datelor de antrenare (overfitting). Penalizându-le, modelul e împins spre soluții mai simple, care de obicei generalizează mai bine pe date nevăzute.

---

## 11. Learning Rate Scheduling

Rata de învățare (learning rate) controlează cât de mari sunt pașii de actualizare a greutăților la fiecare iterație. O rată fixă e rareori optimă pe parcursul întregii antrenări:

- La început, o rată mai mare ajută la explorarea rapidă a spațiului de soluții.
- Spre final, o rată prea mare poate face modelul să "sară" peste minimul optim, fără să se stabilizeze.

Un **scheduler** ajustează automat rata de învățare pe parcursul antrenării după o regulă predefinită (de exemplu, scădere graduală, scădere în trepte, sau o curbă continuă tip cosinus). Scopul general e să combine explorare rapidă la început cu rafinare fină spre final.

---

## 12. Validare încrucișată stratificată (Stratified K-Fold Cross-Validation)

### Problema pe care o rezolvă

Dacă antrenezi și evaluezi pe o singură împărțire train/validare, rezultatul poate fi influențat de "noroc" — acea împărțire particulară poate fi nereprezentativă.

### K-Fold Cross-Validation

Împarte datele în K subseturi ("folduri"). Antrenezi K modele, fiecare folosind K-1 folduri pentru antrenare și 1 fold (diferit de fiecare dată) pentru validare. Asta dă o estimare mai robustă a performanței, pentru că fiecare exemplu ajunge la un moment dat în setul de validare.

### Partea "Stratified"

Stratificarea garantează că fiecare fold păstrează **aceeași proporție de clase** ca în setul original. Fără stratificare, la un set de date dezechilibrat (unele clase mai rare decât altele), unele folduri ar putea ajunge cu foarte puține exemple (sau deloc) dintr-o clasă rară, ceea ce ar distorsiona atât antrenarea cât și evaluarea.

### Ensemble de modele din K-Fold

Un beneficiu suplimentar: la final poți avea K modele antrenate pe subseturi ușor diferite de date. Combinând predicțiile lor (ensemble/voting), reduci varianța predicției finale — erorile individuale ale unui model tind să fie compensate de celelalte.

---

## 13. Data Augmentation

Tehnica de a genera variante ușor modificate ale datelor de antrenare (translații, rotații, zgomot, etc.), aplicate **doar la antrenare**, niciodată la validare/test.

De ce ajută:

- Mărește efectiv diversitatea datelor văzute de model, fără să ai nevoie de date noi reale.
- Forțează modelul să învețe trăsături **invariante** la acele transformări (ex. dacă translatezi puțin imaginea și clasa rămâne aceeași, modelul învață că poziția exactă nu definește clasa).
- Reduce overfitting — modelul nu mai poate "memora" exact pixelii exemplelor de antrenare, pentru că aceștia variază ușor de la o epocă la alta.

### Zgomot (noise injection)

Adăugarea de zgomot aleator peste date simulează imperfecțiuni/variabilitate realistă și face modelul mai robust la mici perturbații care ar putea apărea și în date noi, nevăzute.

---

## 14. Test-Time Augmentation (TTA)

Aplică aceeași idee de augmentare, dar **la inferență**, nu la antrenare: în loc să faci o singură predicție pe imaginea originală, faci mai multe predicții pe variante ușor modificate ale aceleiași imagini (ex. translatată stânga/dreapta), apoi combini (de obicei prin medie ponderată) toate aceste predicții într-una singură.

De ce ajută:

- Reduce varianța predicției finale — o singură predicție poate fi sensibilă la mici detalii ale acelei versiuni exacte a imaginii; media mai multor versiuni tinde să fie mai stabilă.
- E un fel de "ensemble" aplicat unei singure imagini, în loc de mai multe modele.
- Costă timp suplimentar de calcul la inferență (de N ori mai multe forward pass-uri), deci e un compromis explicit între acuratețe și viteză.

---

## 15. Ensemble de modele (Model Voting / Committee)

Ideea generală: în loc să te bazezi pe predicția unui singur model, combini predicțiile mai multor modele (antrenate separat, eventual pe date sau inițializări diferite).

De ce funcționează:

- Modelele diferite tind să greșească în moduri diferite (erori necorelate). Combinând predicțiile lor (medie, vot majoritar, etc.), erorile individuale au șanse mai mari să se anuleze reciproc.
- E un caz particular al unui principiu statistic mai larg: media mai multor estimatori cu erori independente are, de obicei, varianță mai mică decât oricare estimator individual.

---

## Rezumat — fluxul conceptual al unui CNN tipic

1. **Convoluții** extrag trăsături locale (margini, texturi, apoi pattern-uri din ce în ce mai complexe pe măsură ce rețeaua e mai adâncă).
2. **Batch Norm + activări neliniare** stabilizează antrenarea și introduc capacitatea de a învăța relații neliniare.
3. **Pooling / stride** reduc treptat dimensiunea spațială, crescând câmpul receptiv și reducând costul de calcul.
4. **Conexiunile reziduale** permit rețele mai adânci fără degradarea gradientului.
5. **Attention (ex. SE)** ajută rețeaua să se concentreze pe canalele/trăsăturile cele mai relevante.
6. **Global pooling** transformă harta spațială finală într-un vector compact, independent de rezoluția de intrare.
7. **Dropout + weight decay + label smoothing** sunt tehnici de regularizare care combat overfitting-ul din unghiuri diferite.
8. **Cross-validation, augmentare, TTA și ensemble** sunt tehnici la nivel de proces de antrenare/evaluare (nu de arhitectură) care cresc robustețea și generalizarea rezultatului final.
