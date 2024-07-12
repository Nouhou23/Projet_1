import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt 

import sys
sys.path.append('/home/nouhou/Nouhou_repertoire/rhc_dataset-main') 

from multi_label_vae_module import ModularVAE



class EvaluationLatenteY(nn.Module):
    
    def __init__(self, input_dim=1024, latent_dim=100, hidden_dim=4096, num_labels1=10, num_labels2=4, num_labels3=2, output = 2, use_labels_in_input=False):
        super(EvaluationLatenteY, self).__init__()
         
        """
        input_dim: la dimension des features à l'entrée, ici 32*32=1024;
        hidden_dim : nombre de couches cachées (choisi au hasard);  
        latent_dim: dimension de l'espace latent (choisi au hasard);
        num_labels: nombre de categories dans chaque cible;
        use_labels_in_input : option pour utiliser les cibles dans l'encodeur;
        output: sorti  pour la couche fc2 car num_labels3=1 pendant l'entrainement de ModularVAE """
        
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3
        self.hidden_dim = hidden_dim
        self.output = output 
        self.use_labels_in_input = use_labels_in_input
        
        
        ### les couches de sorti pour les cibles 
        self.fc1 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_labels1)
        self.fc3 = nn.Linear(hidden_dim, self.num_labels2)
        self.fc4 = nn.Linear(hidden_dim, self.output)
        
    def prediction_labels(self, x):
        
        """A partir du ModularVAE, on recupère l'espace latent afin d'effectuer la prediction avec softmax"""
        modular_vae = ModularVAE(self.input_dim,  self.hidden_dim, self.latent_dim, self.num_labels1, self.num_labels2, self.num_labels3, use_labels_in_input=False)
        mean, logvar = modular_vae.encode(x)
        latent = modular_vae.reparameterize(mean, logvar)
        h = F.relu(self.fc1(latent))
        
        return F.softmax(self.fc2(h), dim=1), F.softmax(self.fc3(h), dim=1), F.softmax(self.fc4(h), dim=1)
    
    def forward(self, x):
        
        y_hat1, y_hat2,y_hat3 = self.prediction_labels(x)
        return y_hat1, y_hat2, y_hat3 
    
    def loss_function(self, y_hat1, y_hat2, y_hat3, y1, y2, y3):
        
        criterion = nn.CrossEntropyLoss()
        
        y1 = y1.squeeze().long()
        y2 = y2.squeeze().long()
        y3 = y3.squeeze().long()
    
        loss1 = criterion(y_hat1, y1)
        loss2 = criterion(y_hat2, y2)
        loss3 = criterion(y_hat3, y3)
        
        return loss1 + loss2 + loss3
    
    def plot_loss(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    

    def train_module(self, train_loader, optimizer, epochs):
        losses =[]
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch_idx, (x, y1, y2, y3) in enumerate(train_loader):
                optimizer.zero_grad()
                
                y_hat1, y_hat2 , y_hat3 = self.forward(x)
                
                loss = self.loss_function(y_hat1, y_hat2 ,y_hat3, y1, y2, y3)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                losses.append(loss.item()) 
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss /len(train_loader)}')
            
        self.plot_loss(losses)
    
    def _print_metrics(self, labels, preds, label_name):
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted',zero_division=0)
        recall = recall_score(labels, preds, average='weighted',zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(labels, preds)
        class_report = classification_report(labels, preds, zero_division=0)
    
        metrics = [
            ['Accuracy', accuracy],
            ['Precision', precision],
            ['Recall', recall],
            ['F1 Score', f1]
        ]
        print(f"Metriques pour {label_name}:")
        print(tabulate(metrics, headers=['Metriques', 'Valeurs'], tablefmt='pretty'))
        print(f"\nMatrice de confusion pour {label_name}:")
        print(tabulate(conf_matrix, tablefmt='pretty'))
        print(f"\nRapport de classification pour {label_name}:")
        print(class_report)
            
    def evaluation(self, test_loader):
        all_labels1, all_preds1 = [], []
        all_labels2, all_preds2 = [], []
        all_labels3, all_preds3 = [], []
     
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, y1, y2, y3) in enumerate(test_loader):
                y_hat1, y_hat2, y_hat3 = self(x)
                
                _, predicted1 = torch.max(y_hat1.detach(), 1)
                _, predicted2 = torch.max(y_hat2.detach(), 1)
                _, predicted3 = torch.max(y_hat3.detach(), 1)
             
                all_labels1.extend(y1.cpu().numpy())
                all_preds1.extend(predicted1.cpu().numpy())
                
                all_labels2.extend(y2.cpu().numpy())
                all_preds2.extend(predicted2.cpu().numpy())
                
                all_labels3.extend(y3.cpu().numpy())
                all_preds3.extend(predicted3.cpu().numpy())
     
        self._print_metrics(all_labels1, all_preds1, 'y1')
        self._print_metrics(all_labels2, all_preds2, 'y2')
        self._print_metrics(all_labels3, all_preds3, 'y3')
    
   
    
   
class EvaluationFromX(nn.Module):
    
    def __init__(self, input_dim=1024, hidden_dim=4096, num_labels1=10, num_labels2=4, output=2):
        super(EvaluationFromX, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.output = output
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.num_labels1)
        self.fc3 = nn.Linear(hidden_dim, self.num_labels2)
        self.fc4 = nn.Linear(hidden_dim, self.output)
        
    def prediction_targets(self, x):
        h = F.relu(self.fc1(x))
        return F.softmax(self.fc2(h), dim=1), F.softmax(self.fc3(h), dim=1), F.softmax(self.fc4(h), dim=1)
    
    def forward(self, x):
        y_hat1, y_hat2, y_hat3 = self.prediction_targets(x)
        return y_hat1, y_hat2, y_hat3
    
    def fonction_de_perte(self, y_hat1, y_hat2, y_hat3, y1, y2, y3):
        criterion = nn.CrossEntropyLoss()
        
        y1 = y1.squeeze().long()
        y2 = y2.squeeze().long()
        y3 = y3.squeeze().long()
        
        perte1 = criterion(y_hat1, y1)
        perte2 = criterion(y_hat2, y2)
        perte3 = criterion(y_hat3, y3)
        return perte1 + perte2 + perte3
    
    def plot_loss(self, losses):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def train_evaluation(self, train_loader, optimizer, epoch):
        losses = []
        for epo in range(epoch):
            self.train()
            total_loss = 0
            for batch_idx, (x, y1, y2, y3) in enumerate(train_loader):
                optimizer.zero_grad()
                y_hat1, y_hat2, y_hat3 = self.forward(x)
                loss = self.fonction_de_perte(y_hat1, y_hat2, y_hat3, y1, y2, y3)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                losses.append(loss.item())  
                
            print(f'Epoch [{epo+1}/{epoch}], Loss: {total_loss / len(train_loader)}')
            
        self.plot_loss(losses)            
            
    def _print_metrics(self, labels, preds, label_name):
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted',zero_division=0)
        recall = recall_score(labels, preds, average='weighted',zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(labels, preds)
        class_report = classification_report(labels, preds, zero_division=0)
    
        metrics = [
            ['Accuracy', accuracy],
            ['Precision', precision],
            ['Recall', recall],
            ['F1 Score', f1]
        ]
        print(f"Metriques pour {label_name}:")
        print(tabulate(metrics, headers=['Metriques', 'Valeurs'], tablefmt='pretty'))
        print(f"\nMatrice de confusion pour {label_name}:")
        print(tabulate(conf_matrix, tablefmt='pretty'))
        print(f"\nRapport de classification pour {label_name}:")
        print(class_report)
            
    def evaluation(self, test_loader):
        all_labels1, all_preds1 = [], []
        all_labels2, all_preds2 = [], []
        all_labels3, all_preds3 = [], []
     
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, y1, y2, y3) in enumerate(test_loader):
                y_hat1, y_hat2, y_hat3 = self(x)
                
                _, predicted1 = torch.max(y_hat1.detach(), 1)
                _, predicted2 = torch.max(y_hat2.detach(), 1)
                _, predicted3 = torch.max(y_hat3.detach(), 1)
             
                all_labels1.extend(y1.cpu().numpy())
                all_preds1.extend(predicted1.cpu().numpy())
                
                all_labels2.extend(y2.cpu().numpy())
                all_preds2.extend(predicted2.cpu().numpy())
                
                all_labels3.extend(y3.cpu().numpy())
                all_preds3.extend(predicted3.cpu().numpy())
     
        self._print_metrics(all_labels1, all_preds1, 'y1')
        self._print_metrics(all_labels2, all_preds2, 'y2')
        self._print_metrics(all_labels3, all_preds3, 'y3')