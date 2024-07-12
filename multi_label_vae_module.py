import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt




class ModularVAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=4096, latent_dim=100, num_labels1=10, num_labels2=4, num_labels3=2, use_labels_in_input=False):
              
        super(ModularVAE, self).__init__()
                    
        """
        input_dim: la dimension des features à l'entrée, ici 32*32=1024;
        hidden_dim : nombre de couches cachées (choisi au hasard);  
        latent_dim: dimension de l'espace latent (choisi au hasard);
        num_labels: nombre de categories dans chaque cible;
        use_labels_in_input : option pour utiliser les cibles dans l'encodeur """
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_labels1 = num_labels1
        self.num_labels2 = num_labels2
        self.num_labels3 = num_labels3
        self.use_labels_in_input = use_labels_in_input
        
        if use_labels_in_input:
            
            self.full_input_dim = input_dim + num_labels1 + num_labels2 + num_labels3
        else:
            self.full_input_dim = input_dim
        
        # Initialisation des couches pour l'encodeur
             
        # hidden_dim+2*11*11=300
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4, stride=3, padding=1)
        # self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        # self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.fc1 = nn.Linear(self.full_input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialisation des couches pour le décodeur
        self.fc3 = nn.Linear(latent_dim + num_labels1 + num_labels2 + num_labels3, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
                          
    def encode(self, x):
        
        """ À partir de x, l'encodeur produit la distribution des moyennes et des logarithmes des variances
        afin de capturer l'essence des informations dans l'espace latent. (x -> h -> mean et logvar) """
        # h = F.relu(self.conv1(x))
        
        h = F.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        
        """La réparamétrisation est utilisée pour résoudre les problèmes d'échantillonnage à l'entrée de x"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        
        """ Le décodeur reconstruit x à partir de l'espace latent (z -> h -> reconstruction)."""  
        
        h = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h))
        return reconstruction
    
    def forward(self, x, y1, y2, y3):
        
        """ Si l'option use_labels_in_input est activée, alors on transforme les classes de y en encodage one-hot. 
        Sinon, on utilise les caractéristiques pour reconstruire x, régénérer les moyennes (mean) et les logarithmes 
        des variances (logvar), ainsi que les prédictions des sorties."""
        
        ## porblème de dimension : (128x21 and 22x300) =( (batch_size*dim( y1+y3+z) dim(z+num_labels1+num_labels2)* hidden_dim )= (128*(10+1+10)  et  (10+10+2)*300)
       
        y1 = F.one_hot(y1.long().squeeze(1), num_classes=self.num_labels1).float()
        y2 = F.one_hot(y2.long().squeeze(1), num_classes=self.num_labels2).float()
       
        
        if self.use_labels_in_input:
            
            full_input = torch.cat((x, y1, y2, y3), dim=1)
            
        else:
            
            full_input = x
        
        # Encodeur 
        mean, logvar = self.encode(full_input)
        
        # réechantillonnage 
        z = self.reparameterize(mean, logvar)
        
        combined_z_y = torch.cat((z, y1, y2, y3), dim=1)
        
        # construction à partir de l'espace lante et y 
        reconstruction = self.decode(combined_z_y)
        
        return reconstruction, mean, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1):
        
        #### MSE ? 
        
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (beta * KLD + MSE).mean()
        return loss
    
    def plot_loss(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()              
    
    def train_modular_vae(self, train_loader, optimizer, epochs, beta):
        self.train()
        losses = []
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for batch_idx, (x, y1, y2, y3) in enumerate(train_loader):
                optimizer.zero_grad()
                recon_x, mean, logvar = self.forward(x, y1, y2, y3)
                loss = self.loss_function(recon_x, x, mean, logvar, beta=beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                losses.append(loss.item())         
            print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset)}')
            
        self.plot_loss(losses)
        
    def evaluate_modular_vae(self, data_loader, beta=1):
        self.eval()
        total_loss = 0
        losses = []
        with torch.no_grad():
            for batch_idx, (x, y1, y2, y3) in enumerate(data_loader):
                recon_x, mean, logvar = self.forward(x, y1, y2, y3)
                loss = self.loss_function(recon_x, x, mean, logvar, beta=beta)
                total_loss += loss.item()
                losses.append(loss.item()) 
        average_loss = total_loss / len(data_loader.dataset)
        self.plot_loss(losses)
        print(f'Evaluation Loss: {average_loss}')
        return average_loss
