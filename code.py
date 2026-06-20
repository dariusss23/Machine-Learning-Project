import os   # folosit pentru lucrul cu fișiere
import random   # folosit pentru valori aleatoare
import pandas as pd # importa biblioteca pandas cu aliasul pd. Folosita pentru citirea fisierelor CSV
import numpy as np  # nu il folosesc direct in cod
import torch # biblioteca principala folosita pentru tensorii, GPU-ul, salvarea modelelor etc.
import torch.nn as nn   # import modulul retelei neuronale din PyTorch
import torch.nn.functional as F # importa functii utile pentru retele neuronale
from torch.utils.data import Dataset, DataLoader    # import 2 clase importante : Dataset - defineste cum se citeste un exemplu din setul de date, 2 - imparte dataset-ul inbatch-uri pentru antrenare
from torchvision.io import read_image, ImageReadMode    # import functie citire imagini : read_image - citeste imagineaca tensor PyTorch, ImageReadMode.GRAY - imagine citita grayscale, un singur canal
from sklearn.model_selection import StratifiedKFold # StratifiedKFold - impartirea datelor in 5 fold-uri

class CustomImageDataset(Dataset):  # mosteneste Dataset deci PyTorch stie cum sa foloseasca in DataLoader
    def __init__(self, tabel_date, folder_imagini, e_test=False, augmentare=False):
        self.tabel_date=tabel_date  # numele imaginilor si, daca nu suntem in test, etichetele lor
        self.folder_imagini=folder_imagini  # folderul din care citim imaginile
        self.e_test=e_test # e_test=True inseamna ca nu avem etichete, ci doar id-uri de imagini
        self.augmentare=augmentare  # augmentare=True inseamna ca aplicam transformari doar pe datele de antrenare

    def __len__(self):
        return len(self.tabel_date) # daca train.csv are 10.000 de imagini, atunci len(dataset) va fi 10.000.

    def __getitem__(self, index):
        nume_poza=self.tabel_date.iloc[index, 0]    # ia numele imaginii de pe rândul index, coloana 0
        cale_poza=os.path.join(self.folder_imagini, nume_poza) # construim calea completa catre imagine

        # citim imaginea in format grayscale, deoarece spectrogramele au un singur canal relevant
        # si o convertim imaginea la float32 pentru a putea face operatii matematice pe ea
        poza_citita=read_image(cale_poza, mode=ImageReadMode.GRAY)  # [1, H, W]
        # pixelii pot fi numere intregi intre 0 și 255. 
        # pentru retele neuronale avem nevoie de valori reale, ca sa poti face operatii matematice si gradient.
        poza=poza_citita.type(torch.float32)

        # augmentarea se aplica doar la antrenare
        # mutam imaginea pe orizontala cu un numar mic de pixeli
        # ideea este ca modelul sa nu invete doar pozitii fixe
        # zgomot gaussian ptr un model mai robust
        if self.augmentare:
            mutare=random.randint(-6, 6)    # alege un numar aleator intre -6 si 6.
            if mutare!=0:
                poza=torch.roll(poza, shifts=mutare, dims=2)    # dim=0 (canal), dim=1(inaltime/vertical), dim=2 (latime/orizontal)

            zgomot=torch.randn_like(poza) # creeaza zgomot gaussian cu aceeasi forma ca imaginea
            zgomot_scalat=zgomot * 0.03 # micsorez zgomotul
            poza=poza + zgomot_scalat   # adauga zgomot peste imagine

        # media si variatia imaginii curente + normalizare
        medie=poza.mean()
        deviatie=poza.std()
        
        if deviatie > 0:
            diferenta=poza - medie
            poza=diferenta / deviatie
        else:
            poza=poza - medie

        if self.e_test:
            return poza, nume_poza
        else:
            eticheta=self.tabel_date.iloc[index, 1] # iau eticheta de pe coloana 1
            eticheta_tensor=torch.tensor(eticheta, dtype=torch.long) # transforma eticheta intr-un tensor PyTorch de tip long, 
                                                                     # CrossEntropyLoss cere ca etichetele sa fie long, nu float
            return poza, eticheta_tensor

class BlocRezidual(nn.Module):  # definesc un bloc rezidual, asemanator cu ideea din ResNet
    def __init__(self, in_channels, out_channels, pas=1):   # stride-ul primei convolutii
        super().__init__()  # initializez corect clasa parinte nn.Module, fara asta, PyTorch nu ar gestiona corect straturile.
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=pas, padding=1, bias=False)
        # prima convolutie din bloc, kernel 3x3, stride=pas (reduce dim spatiala daca pas>1), padding=1 pastreaza dim imagine daca stride=1, bias=False(dupa conv ai BN, iar acesta devine mai putin necesar)
        self.bn1=nn.BatchNorm2d(out_channels)   # batch normalization dupa prima convolutie, stabilizeaza antrenarea, modelul inv mai bn
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        # daca se schimba dimensiunea spatiala sau numarul de canale,
        # trebuie sa aliniem intrarea pentru a putea face adunarea reziduala
        self.scurtatura=nn.Sequential() # initializez shortcut-ul ca o secventa goala
        if pas != 1 or in_channels != out_channels:
            strat_conv_scurtatura=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=pas, bias=False)
            # creez o conv 1x1, aceasta schimba in_channels in out_channels
            strat_bn_scurtatura=nn.BatchNorm2d(out_channels)    # BN ptr shortcut
            
            # shortcut-ul devine conv 1x1 + BN
            self.scurtatura=nn.Sequential(
                strat_conv_scurtatura,
                strat_bn_scurtatura
            )

    def forward(self, x):
        # trecem intrarea prin prima convolutie + batch norm + activare
        iesire_conv1=self.conv1(x)
        iesire_bn1=self.bn1(iesire_conv1)
        rezultat=F.leaky_relu(iesire_bn1, 0.1)

        # trecem mai departe prin a doua convolutie + batch norm
        iesire_conv2=self.conv2(rezultat)
        iesire_bn2=self.bn2(iesire_conv2)
        rezultat=iesire_bn2

        # adaugam shortcut-ul peste iesirea principala
        iesire_scurtatura=self.scurtatura(x)
        rezultat=rezultat + iesire_scurtatura
        # IDEE : modelul invata o corectie peste intrarea initiala, nu trebuie să reinventeze totul de la zero

        # activare finala bloc
        rezultat_final=F.leaky_relu(rezultat, 0.1)
        
        return rezultat_final

class NeuralNetwork(nn.Module): # reteaua CNN cu blocuri residuale
    def __init__(self):
        super().__init__()  # initialize modelul pytorch
        
        # creez primul bloc de procesare ca o secventa de straturi
        # stratul initial extrage primele trasaturi din imagine
        # kernelul este mai lat pe orizontala pentru a surprinde mai bine structurile semnalului
        self.strat_initial=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 7), padding=(1, 3), bias=False),
            # 1 canal -> 64 canale, 3 - orizontala, 7 - verticala
            nn.BatchNorm2d(64), # normalizez cele 64 canale
            nn.LeakyReLU(0.1)
        )

        # cele 4 blocuri reziduale cresc treptat numarul de canale  
        self.bloc1=BlocRezidual(64, 64)
        self.pool1=nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # kernel(1, 2) vertical nu comprima, orizontal comprima

        self.bloc2=BlocRezidual(64, 128)
        self.pool2=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # reduce si inaltimea si latimea la jumătate

        self.bloc3=BlocRezidual(128, 256)
        self.pool3=nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.bloc4=BlocRezidual(256, 512)
        
        # acest pooling transforma fiecare harta de trasaturi intr-o singura valoare
        self.gap=nn.AdaptiveAvgPool2d((1, 1))
        # [batch, 512, H, W] -> [batch, 512, 1, 1]

        # clasificatorul final produce scoruri pentru cele 5 clase posibile
        self.decizie_finala=nn.Sequential(
            nn.Flatten(),   # transforma tensorul [batch, 512, 1, 1] -> [batch, 512]
            nn.Linear(512, 128), # fully connected 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4), # opresc 40% neuroni ptr Overfitting
            nn.Linear(128, 5)
        )

    def forward(self, poza):
        poza_initiala=self.strat_initial(poza)
        
        # blocurile reziduale + pooling

        iesire_bloc1=self.bloc1(poza_initiala)
        poza_pool1=self.pool1(iesire_bloc1)

        iesire_bloc2=self.bloc2(poza_pool1)
        poza_pool2=self.pool2(iesire_bloc2)

        iesire_bloc3=self.bloc3(poza_pool2)
        poza_pool3=self.pool3(iesire_bloc3)

        iesire_bloc4=self.bloc4(poza_pool3)
        
        # comprimam reprezentarea
        poza_gap=self.gap(iesire_bloc4)

        # scorurile finale ptr fiecare clasa
        rezultat=self.decizie_finala(poza_gap)
        
        return rezultat

print("Incarc date antrenament...")
tabel_total=pd.read_csv('train.csv')

# etichetele initiale sunt 1, 2, 3, 4, 5
# pentru CrossEntropyLoss trebuie sa fie 0, 1, 2, 3, 4
tabel_total['label']=tabel_total['label'] - 1

# alegem automat GPU daca exista, altfel CPU
device="cuda" if torch.cuda.is_available() else "cpu"

NUM_MODELE=5
NUM_EPOCHS=35

# impartim datele in 5 folduri stratificate, astfel incat distributia claselor sa se pastreze
impartitor_date=StratifiedKFold(n_splits=NUM_MODELE, shuffle=True, random_state=42)
# random_state=42 (impartirea este reproductibila)

for numar_model, (indecsi_antrenament, indecsi_validare) in enumerate(impartitor_date.split(tabel_total, tabel_total['label'])):
    print()
    print(f"=== Antrenez modelul {numar_model + 1} din {NUM_MODELE} ===")

    # selectam datele de antrenare pentru foldul curent
    date_antrenament_brute=tabel_total.iloc[indecsi_antrenament]
    date_antrenament=date_antrenament_brute.reset_index(drop=True) # resetez index tabel, fara asta ar avea indexuri vechi din tabelul original

    # selectam datele de validare pentru foldul curent
    date_validare_brute=tabel_total.iloc[indecsi_validare]
    date_validare=date_validare_brute.reset_index(drop=True)

    # construim dataset-urile
    set_antrenament=CustomImageDataset(date_antrenament, 'train/', augmentare=True)
    set_validare=CustomImageDataset(date_validare, 'train/', augmentare=False)

    # construim dataloader-ele
    # modelul vede cate 64 de imagini odata, sunt amestecate la fiecare epoca
    train_dataloader=DataLoader(set_antrenament, batch_size=64, shuffle=True)
    test_dataloader=DataLoader(set_validare, batch_size=64) # Nu am shuffle = True la validare nu conteaza ordinea

    # initializam modelul pentru foldul curent si il punem sa ruleze pe CPU/GPU
    model=NeuralNetwork().to(device)

    # functia de pierdere pentru clasificare multi-clasa, label_smoothing reduce supra-increderea modelului
    # Adam este optimizatorul folosit pentru actualizarea ponderilor weight_decay=1e-4 este regularizare L2, ajuta contra overfitw
    # scheduler-ul scade treptat learning rate-ul pe parcursul antrenarii (la inceput inv mai agresiv, spre final face pasi mici)

    loss_funciton=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizator=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizator, T_max=NUM_EPOCHS)

    cea_mai_buna_acuratete=0.0
    nume_fisier_model=f'model_expert_{numar_model}.pth'

    for epoca in range(NUM_EPOCHS):
        model.train()
        
        # parcurgem toate batch-urile de antrenare
        for image_batch, labels_batch in train_dataloader:
            
            # mut imaginile pe CPU/GPU (trb sa fie pe acelasi dispozitiv ca modelul)
            image_batch=image_batch.to(device)
            labels_batch=labels_batch.to(device)

            predictii=model(image_batch)    # predicita modelului (tensor de scoruri de forma [batch_size, 5])
            eroare=loss_funciton(predictii, labels_batch)   # eroarea fata de etichetele corecte

            optimizator.zero_grad() # stergem gradientii vechi
            eroare.backward()   # calculam gradientii noi
            optimizator.step()  # actualizam parametrii modelului

        #validare fiecare epoca 
        
        raspunsuri_corecte=0

        model.eval() # pun modelul in mod evaluare (opreste Dropout-ul si schimba comportamentul BN-ul)

        with torch.no_grad(): # dezactivez calcul gradienti, la validare nu actualizez modelul, deci nu am nevoie de gradienti
            for image_batch, labels_batch in test_dataloader:
                
                image_batch=image_batch.to(device)
                labels_batch=labels_batch.to(device)

                predictii=model(image_batch)    # predictii pe setul de validare

                raspunsuri_alese=predictii.argmax(1)    # alegem clasa cu scorul cel mai mare
                
                # verificam cate raspunsuri sunt corecte
                comparatie=(raspunsuri_alese == labels_batch)
                suma_corecte=comparatie.sum()
                raspunsuri_corecte += suma_corecte.item() # transforma tensorul intr-un numar python simplu

        # calculam acuratetea pe validare
        total_imagini_validare=len(test_dataloader.dataset)
        acuratete=100 * raspunsuri_corecte / total_imagini_validare

        print(f"Epoca {epoca+1}/{NUM_EPOCHS} | Acuratete: {acuratete:.1f}%")
        scheduler.step()    # actualizam scheduler-ul

        if acuratete > cea_mai_buna_acuratete:
            cea_mai_buna_acuratete=acuratete
            stare_model=model.state_dict() # contine ponderile si bias-urile modelului
            torch.save(stare_model, nume_fisier_model)
            print(f"-> Modelul {numar_model + 1} record nou: {cea_mai_buna_acuratete:.1f}%! Salvat.")

print()
print("=== Incarc toti cei 5 experti si generez fisierul final ===")

# citim fisierul sample_submission pentru a lua id-urile imaginilor de test
tabel_test_kaggle=pd.read_csv('sample_submission.csv')
# construim dataset-ul de test
set_test=CustomImageDataset(tabel_test_kaggle, 'test/', e_test=True, augmentare=False)
# dataloader pentru test
test_kaggle_dataloader=DataLoader(set_test, batch_size=64)

comitet_experti=[]
for i in range(NUM_MODELE):
    model_incarcat=NeuralNetwork().to(device) # creez o instanta noua de model cu aceeasi arhitectura
    cale_salvare=f'model_expert_{i}.pth' # construiesc numele fisierului salvat
    model_incarcat.load_state_dict(torch.load(cale_salvare)) # incarc ponderile salvate in model
    model_incarcat.eval() # pun modelul in modul evaluare
    comitet_experti.append(model_incarcat) # adaug modelul in experti

lista_clase_ghicite=[] # lista pentru clasele prezise
lista_id_uri=[] # lista pentru ID-uri imagini

with torch.no_grad(): # desactivez gradientii pentru ca acum doar fac predictii
    # pargurg imaginile de test in batch-uri
    # image_batch = imaginile
    # id__uri = numele/ID-urile imaginilor
    for image_batch, id_uri in test_kaggle_dataloader:
        image_batch=image_batch.to(device)
        
        # initializam suma voturilor pentru batch-ul curent
        dimensiune_batch=image_batch.size(0) # aflu cate imagini am in batch-ul curent
        suma_voturi=torch.zeros((dimensiune_batch, 5)) # creez un tensor de zerouri pentru voturi (forma : [dimensiune_batch, 5])
        suma_voturi=suma_voturi.to(device)

        # fiecare expert voteaza pentru fiecare imagine
        for expert in comitet_experti:
            vot_normal=expert(image_batch)  # predictia pe imaginea originala

            # creez o imagine mutata la stanga cu 5 pixeli
            # batch de forma : [batch, canal, inaltime, latime], dim=3 (latimea imaginii)
            
            imagine_mutata_stanga=torch.roll(image_batch, shifts=-5, dims=3)
            vot_mutat_stanga=expert(imagine_mutata_stanga)  # predictia pe imaginea mutata putin la stanga cu 5px

            imagine_mutata_dreapta=torch.roll(image_batch, shifts=5, dims=3)
            vot_mutat_dreapta=expert(imagine_mutata_dreapta)    # predictia pe imaginea mutata putin la dreapta

            # TTA (Test-Time Augmentation) - la  un test nu folosesc doar img principala, ci si versiuni usor modificate ale ei
            
            # dam o pondere mai mare imaginii originale si ponderi mai mici versiunilor mutate
            vot_normal_ponderat=vot_normal * 0.5
            vot_stanga_ponderat=vot_mutat_stanga * 0.25
            vot_dreapta_ponderat=vot_mutat_dreapta * 0.25
            
            vot_total_expert=vot_normal_ponderat + vot_stanga_ponderat + vot_dreapta_ponderat
            suma_voturi=suma_voturi + vot_total_expert

        vot_comitet=suma_voturi / NUM_MODELE

        # alegem clasa finala pentru fiecare imagine
        tensor_clase_prezise=vot_comitet.argmax(1)
        # mutam pe CPU pentru a converti in numpy
        tensor_clase_cpu=tensor_clase_prezise.cpu()
        clase_prezise=tensor_clase_cpu.numpy()
        
        # deoarece la antrenare am folosit etichete 0-4,
        # acum revenim la etichetele originale 1-5
        clase_bune_pentru_kaggle=clase_prezise + 1

        lista_clase_ghicite.extend(clase_bune_pentru_kaggle) # adaug predictiile batch-ului in lista finala
        lista_id_uri.extend(id_uri) # adaug ID-urile imaginilor in lista finala

fisier_submisie=pd.DataFrame({ # creez un dataframe cu 2 coloane
    'id': lista_id_uri, 
    'label': lista_clase_ghicite
})
fisier_submisie.to_csv('submisie_kaggle.csv', index=False) # nu adaug coloana de index generata de pandas

print("Fisierul 'submisie_kaggle.csv' salvat")
