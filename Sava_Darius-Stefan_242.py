import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from sklearn.model_selection import StratifiedKFold

class CustomImageDataset(Dataset):
    def __init__(self, tabel_date, folder_imagini, e_test=False, augmentare=False):
        self.tabel_date=tabel_date  # numele imaginilor si, daca nu suntem in test, etichetele lor
        self.folder_imagini=folder_imagini  # folderul din care citim imaginile
        self.e_test=e_test # e_test=True inseamna ca nu avem etichete, ci doar id-uri de imagini
        self.augmentare=augmentare  # augmentare=True inseamna ca aplicam transformari doar pe datele de antrenare

    def __len__(self):
        return len(self.tabel_date)

    def __getitem__(self, index):
        # luam numele imaginii de pe pozitia index
        nume_poza=self.tabel_date.iloc[index, 0]
        cale_poza=os.path.join(self.folder_imagini, nume_poza) # construim calea completa

        # citim imaginea in format grayscale, deoarece spectrogramele au un singur canal relevant
        # si o convertim imaginea la float32 pentru a putea face operatii matematice pe ea
        poza_citita=read_image(cale_poza, mode=ImageReadMode.GRAY)
        poza=poza_citita.type(torch.float32)

        # augmentarea se aplica doar la antrenare
        # mutam imaginea pe orizontala cu un numar mic de pixeli
        # ideea este ca modelul sa nu invete doar pozitii fixe
        # zgomot gaussian ptr un model mai robust
        if self.augmentare:
            mutare=random.randint(-6, 6)
            if mutare!=0:
                poza=torch.roll(poza, shifts=mutare, dims=2)

            zgomot=torch.randn_like(poza)
            zgomot_scalat=zgomot * 0.03
            poza=poza + zgomot_scalat

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
            eticheta=self.tabel_date.iloc[index, 1]
            eticheta_tensor=torch.tensor(eticheta, dtype=torch.long)
            return poza, eticheta_tensor

class BlocRezidual(nn.Module):
    def __init__(self, in_channels, out_channels, pas=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=pas, padding=1, bias=False) # prima convolutie din bloc
        self.bn1=nn.BatchNorm2d(out_channels)   # batch normalization dupa prima convolutie
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        # daca se schimba dimensiunea spatiala sau numarul de canale,
        # trebuie sa aliniem intrarea pentru a putea face adunarea reziduala
        self.scurtatura=nn.Sequential()
        if pas != 1 or in_channels != out_channels:
            strat_conv_scurtatura=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=pas, bias=False)
            strat_bn_scurtatura=nn.BatchNorm2d(out_channels)
            
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

        # activare finala bloc
        rezultat_final=F.leaky_relu(rezultat, 0.1)
        
        return rezultat_final

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # stratul initial extrage primele trasaturi din imagine
        # kernelul este mai lat pe orizontala pentru a surprinde mai bine structurile semnalului
        self.strat_initial=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # cele 4 blocuri reziduale cresc treptat numarul de canale  
        self.bloc1=BlocRezidual(64, 64)
        self.pool1=nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.bloc2=BlocRezidual(64, 128)
        self.pool2=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bloc3=BlocRezidual(128, 256)
        self.pool3=nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.bloc4=BlocRezidual(256, 512)
        
        # acest pooling transforma fiecare harta de trasaturi intr-o singura valoare
        self.gap=nn.AdaptiveAvgPool2d((1, 1))

        # clasificatorul final produce scoruri pentru cele 5 clase posibile
        self.decizie_finala=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
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

for numar_model, (indecsi_antrenament, indecsi_validare) in enumerate(impartitor_date.split(tabel_total, tabel_total['label'])):
    print()
    print(f"=== Antrenez modelul {numar_model + 1} din {NUM_MODELE} ===")

    # selectam datele de antrenare pentru foldul curent
    date_antrenament_brute=tabel_total.iloc[indecsi_antrenament]
    date_antrenament=date_antrenament_brute.reset_index(drop=True)

    # selectam datele de validare pentru foldul curent
    date_validare_brute=tabel_total.iloc[indecsi_validare]
    date_validare=date_validare_brute.reset_index(drop=True)

    # construim dataset-urile
    set_antrenament=CustomImageDataset(date_antrenament, 'train/', augmentare=True)
    set_validare=CustomImageDataset(date_validare, 'train/', augmentare=False)

    # construim dataloader-ele
    train_dataloader=DataLoader(set_antrenament, batch_size=64, shuffle=True)
    test_dataloader=DataLoader(set_validare, batch_size=64)

    # initializam modelul pentru foldul curent
    model=NeuralNetwork().to(device)

    # functia de pierdere pentru clasificare multi-clasa, label_smoothing reduce supra-increderea modelului
    # Adam este optimizatorul folosit pentru actualizarea ponderilor
    # scheduler-ul scade treptat learning rate-ul pe parcursul antrenarii

    loss_funciton=nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizator=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizator, T_max=NUM_EPOCHS)

    cea_mai_buna_acuratete=0.0
    nume_fisier_model=f'model_expert_{numar_model}.pth'

    for epoca in range(NUM_EPOCHS):
        model.train()
        
        # parcurgem toate batch-urile de antrenare
        for image_batch, labels_batch in train_dataloader:
            
            image_batch=image_batch.to(device)
            labels_batch=labels_batch.to(device)

            predictii=model(image_batch)    # predicita modelului
            eroare=loss_funciton(predictii, labels_batch)   # eroarea fata de etichetele corecte

            optimizator.zero_grad() # stergem gradientii vechi
            eroare.backward()   # calculam gradientii noi
            optimizator.step()  # actualizam parametrii modelului

        raspunsuri_corecte=0

        model.eval()

        with torch.no_grad():
            for image_batch, labels_batch in test_dataloader:
                
                image_batch=image_batch.to(device)
                labels_batch=labels_batch.to(device)

                predictii=model(image_batch)    # predictii pe setul de validare

                raspunsuri_alese=predictii.argmax(1)    # alegem clasa cu scorul cel mai mare
                
                # verificam cate raspunsuri sunt corecte
                comparatie=(raspunsuri_alese == labels_batch)
                suma_corecte=comparatie.sum()
                raspunsuri_corecte += suma_corecte.item()

        # calculam acuratetea pe validare
        total_imagini_validare=len(test_dataloader.dataset)
        acuratete=100 * raspunsuri_corecte / total_imagini_validare

        print(f"Epoca {epoca+1}/{NUM_EPOCHS} | Acuratete: {acuratete:.1f}%")
        scheduler.step()    # actualizam scheduler-ul

        if acuratete > cea_mai_buna_acuratete:
            cea_mai_buna_acuratete=acuratete
            stare_model=model.state_dict()
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
    model_incarcat=NeuralNetwork().to(device)
    cale_salvare=f'model_expert_{i}.pth'
    model_incarcat.load_state_dict(torch.load(cale_salvare))
    model_incarcat.eval()
    comitet_experti.append(model_incarcat)

lista_clase_ghicite=[]
lista_id_uri=[]

with torch.no_grad():
    for image_batch, id_uri in test_kaggle_dataloader:
        image_batch=image_batch.to(device)
        
        # initializam suma voturilor pentru batch-ul curent
        dimensiune_batch=image_batch.size(0)
        suma_voturi=torch.zeros((dimensiune_batch, 5))
        suma_voturi=suma_voturi.to(device)

        # fiecare expert voteaza pentru fiecare imagine
        for expert in comitet_experti:
            vot_normal=expert(image_batch)  # predictia pe imaginea originala

            imagine_mutata_stanga=torch.roll(image_batch, shifts=-5, dims=3)
            vot_mutat_stanga=expert(imagine_mutata_stanga)  # predictia pe imaginea mutata putin la stanga

            imagine_mutata_dreapta=torch.roll(image_batch, shifts=5, dims=3)
            vot_mutat_dreapta=expert(imagine_mutata_dreapta)    # predictia pe imaginea mutata putin la dreapta

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

        lista_clase_ghicite.extend(clase_bune_pentru_kaggle)
        lista_id_uri.extend(id_uri)

fisier_submisie=pd.DataFrame({
    'id': lista_id_uri, 
    'label': lista_clase_ghicite
})
fisier_submisie.to_csv('submisie_kaggle.csv', index=False)

print("Fisierul 'submisie_kaggle.csv' salvat")