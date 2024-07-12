import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms , Normalize
import matplotlib.pyplot as plt


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


class DataAugmentationMNIST(Dataset):
    
    """Ce module permet de créer deux cibles catégorielles à partir des images MNIST."""

    def __init__(self, dataset, n_add_rows, n_add_cols, seed=42):
        
        """Initialisation des attributs."""
        
        self.images_tensor = dataset.data.float()
        self.original_targets = dataset.targets.clone()  
        self.to_tensor = ToTensor()
        self.seed = seed

        """Initialisation des attributs utilisés par la fonction images_decalees afin de générer
        une matrice plus grande qu'une image MNIST. n_add_rows :nombre de lignes  à ajouter ;  
        n_add_cols: nombre de colonnes  à ajouter """
        
        self.n_add_rows = n_add_rows
        self.n_add_cols = n_add_cols

    def images_decalees(self):
        
        """Cette fonction récupère d'abord les images MNIST, puis crée une nouvelle matrice remplie de zéros (new_images)
        de taille supérieure à celle de MNIST."""
        
        nrows, ncols = self.images_tensor.shape[1], self.images_tensor.shape[2]
        new_nrows, new_ncols = nrows + self.n_add_rows, ncols + self.n_add_cols
        new_images = torch.zeros((len(self.images_tensor), new_nrows, new_ncols), dtype=self.images_tensor.dtype)
        
        
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(self.images_tensor), generator=generator)
        target_shift = torch.zeros(len(self.images_tensor), dtype=torch.long)
        updated_targets = self.original_targets.clone()  

        for i in range(len(self.images_tensor)):
            
            """On remplit aléatoirement new_images avec une image de MNIST en fonction des valeurs
            aléatoires start_row et start_col dans le but de  créer une nouvelle cible. On recupère égalment 
            la valeur cible de cette  même image """
            
            image_idx = indices[i]
            image = self.images_tensor[image_idx]
            original_target = self.original_targets[image_idx]
            start_row = torch.randint(0, 2, (1,), generator=generator).item() * 2
            start_col = torch.randint(0, 2, (1,), generator=generator).item() * 2
            new_images[i, start_row:start_row + nrows, start_col:start_col + ncols] = image

            if start_row == 0 and start_col == 0:
                
                target = 0
                
            elif start_row == 2 and start_col == 0:
                
                target = 1
                
            elif start_row == 0 and start_col == 2:
                
                target = 2
                
            else:
                
                target = 3

            target_shift[i] = target
            updated_targets[i] = original_target  

        return new_images, target_shift, updated_targets

    def images_inversees(self):
        
        """Cette fonction permet de générer les images inversées."""
        
        image_dec = self.images_decalees()
        new_images, target_shift, updated_targets = image_dec
        image_inverted = abs(new_images - 255)
        
        return image_inverted, new_images, target_shift, updated_targets

    def images_melangees(self):
        
        """Concatenation des images inversées et de new_images ."""
        
        image_inverted, new_images, target_shift, updated_targets = self.images_inversees()
        combined_images = torch.cat((image_inverted, new_images), dim=0)

        """Ici, on crée un tenseur rempli de valeurs 1 pour les images d'origine (new_images) et 
        un tenseur rempli de 0 pour les images inversées, puis on les concatène."""
        
        
        inverted_labels = torch.zeros(len(image_inverted), dtype=torch.long)
        original_labels = torch.ones(len(new_images), dtype=torch.long)
        combined_labels = torch.cat((inverted_labels, original_labels), dim=0)

        """Cette ligne de code permet d'obtenir les étiquettes d'origine de même longueur que combined_labels."""
         
        combined_targets = torch.cat((updated_targets, updated_targets), dim=0)
        indices = torch.randperm(combined_images.size(0), generator=torch.Generator().manual_seed(self.seed))
        
        
        """Ensuite, on récupère le melange des images  entre les images décalées et celles inversées ; 
        on effectue la même opération avec les cibles en attribuant 1 aux images d'origine et 0 aux images inversées."""
        
    
        shuffled_images = combined_images[indices][: len(self.images_tensor)]
        shuffled_target = combined_labels[indices][: len(self.images_tensor)]
        shuffled_updated_targets = combined_targets[indices][: len(self.images_tensor)]
        shuffled_target_shift = target_shift[indices % len(target_shift)][: len(self.images_tensor)]
        
        
        return shuffled_images, shuffled_updated_targets, shuffled_target_shift, shuffled_target

  
class Convertisseur:
       
    def __init__(self, dataset, dim_input):
        
        """ dim_input : il s'agit de la dimension de la matrice, par exemple, pour une matrice de 32x32, dim_input serait égal à 1024."""

        self.x = dataset[0]
        self.y = dataset[1:]
        self.dim_input = dim_input
    
    def diviseur_reshape(self):
       
        x = self.x / 255.0
        composer = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])
        x = composer(x)

        x = x.view(-1, self.dim_input) 
        
        y_reshape = torch.zeros((len(self.y), len(self.y[0]), 1), dtype=torch.float)
        
        for i in range(len(self.y)):
            
            y_tensor_res = self.y[i].reshape(-1, 1)
            y_reshape[i] = y_tensor_res
        
        return x, y_reshape
           
              
# data_train = DataAugmentationMNIST(train_dataset,4,4).images_melangees()

# convertisseur = Convertisseur(data_train, 1024)

# x_train_tensor, y_reshape = convertisseur.diviseur_reshape()

# x_train_tensor.size()

# y_reshape[0].shape


# #######################################################################################################
# ######################################################################################################
# #####################################################################################################             
     

# data_train = DataAugmentationMNIST(train_dataset, 4, 4).images_melangees()

# data_test = DataAugmentationMNIST(test_dataset, 4, 4).images_melangees()



# new_images = data_train[0][:20]

# fig, axes = plt.subplots(2, 10, figsize=(20, 6))  
# for i in range(10):
#     axes[0, i].imshow(new_images[i].numpy(), cmap='gray')  
#     axes[0, i].set_title(f'Image {i+1}')
#     axes[0, i].axis('off')

# for i in range(10):
#     axes[1, i].imshow(new_images[10+i].numpy(), cmap='gray')  
#     axes[1, i].set_title(f'Image {10+i+1}')
#     axes[1, i].axis('off')

# plt.tight_layout()  
# plt.show()

# data_train[1][:20]

# data_train[3][:20]

# data_train[2][:20]

# t =data_test[3]
# y ,x =torch.unique(t, return_counts=True)

# y ,x


# data_train[0].shape

# data_train[1].shape

# data_train[2].shape

# data_train[3].shape

# #######################################################
# #####################################################


# new_images = data_test[0][:20]

# fig, axes = plt.subplots(2, 10, figsize=(20, 6))  
# for i in range(10):
#     axes[0, i].imshow(new_images[i].numpy(), cmap='gray')  
#     axes[0, i].set_title(f'Image {i+1}')
#     axes[0, i].axis('off')

# for i in range(10):
#     axes[1, i].imshow(new_images[10+i].numpy(), cmap='gray')  
#     axes[1, i].set_title(f'Image {10+i+1}')
#     axes[1, i].axis('off')

# plt.tight_layout()  
# plt.show()



# data_test[1][:20]

# data_test[3][:20]

# data_test[2][:20]



# t =data_test[2]
# y ,x =torch.unique(t, return_counts=True)

# y ,x



# data_test[0].shape

# data_test[1].shape

# data_test[2].shape

# data_test[3].shape

