#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Martes 3 marzo de 2020
@author: sergio
"""
# ================================================================================================================
#  Modules
# ================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
#  Funcion Principal, que muestra el mapa de color.
# =============================================================================










            

# ================================================================================================================
#                           Color Map Class
# ================================================================================================================
class Color_Map():
    '''Esta Clase Crea una imagen  RGB (un mapa de color)  de una IMagen en 2D
      Cuyas entradas  enteras que identifican a cada pixel a una clase '''
    def __init__(self, Image_Class,Color_Palette = None):
        self.Image_Class = Image_Class.astype(int)                            # Se guarda la Imagen para trabajar con ella
        self.rows,self.columns =  Image_Class.shape                           # Dimension Espacial de la Imagen
        self._class_ = np.unique(self.Image_Class)                            # Aqui se define el vector de clases
        self.n_class = self._class_.shape[0]                                  # Aqui se calcula el numero de clases  
        if Color_Palette is not None:
            self.color_palette = Color_Palette
        else:
            import seaborn as sns  # Este Modulo Me permite llamar una paleta de colores
            mydict = {}
            Palette = np.array(sns.color_palette("hls", self.n_class))
            cont = int(0)
            for i in self._class_:
                mydict[i] = ['Class'+str(i),Palette[cont]]
                cont = cont +1
            self.color_palette = mydict
        self.Img_Color = self._ImageRGB()                                     # Se Crea la Imagen de Color  
        
    def _ImageRGB(self): 
        Img_Color =  np.zeros([self.rows,self.columns,3])
        for i in range(0,self.rows):
            for j in range(0,self.columns):
                for x,y in self.color_palette.items():
                    if self.Image_Class[i,j] == x:
                        Img_Color[i,j,:] = y[1]
                        break
                    else: continue
        return Img_Color
    
    def Get_ColorMapImage(self):
        return self.Img_Color
        
    def Get_ColorPalete(self):
        return self.color_palette
    
    def Get_ClasesColorPallete(self):
        out = []
        for i,j in self.color_palette.items():
            if i != -1:
                out.append(j[1])
        return out
    
    def Get_ClasesNamePallete(self):
        out = []
        for i,j in self.color_palette.items():
            if i != -1:
                out.append(j[0])
        return out

    def Plot(self,axes,Title = 'None'):
        # Asegurarse importar import matplotlib.pyplot as plt
        axes.imshow(self.Img_Color)
        axes.axis('off')
        if Title is not None:
            axes.set_title(Title,fontsize=25)
   
    @classmethod
    def _imshowMap2(cls,Img,Gt,Color_Palette,title=None,subtitleIMG =None,subtitleGt = None):
         # import matplotlib.pyplot as plt
         ColorMap_True  =   cls(Gt,Color_Palette = Color_Palette)      # Creando el Objeto Color Map
         ColorMap_Pred  =   cls(Img,Color_Palette = Color_Palette)       # Creando el Objeto Color Map
         fig,axes  = plt.subplots(nrows =1,
                             ncols =3,
                             figsize=(50,23),
                             sharex='none',
                             sharey='none')   #img_shape[0]/50,img_shape[1]/50
         fig.subplots_adjust(wspace=0.1,hspace=0.15)
         
         if title ==None:
             fig.suptitle('Original vs Predicted',fontsize=20)
         else:
             fig.suptitle(title,fontsize=20)
             
         axes = axes.ravel()  # convert directions of axes to array of dimention 1
         # Ground Truth
         if subtitleGt is None:
             ColorMap_True.Plot(axes[0],'Ground Truth')
         else:
             ColorMap_True.Plot(axes[0],subtitleGt)
         # Predicted Image
         if subtitleIMG is None:
             ColorMap_Pred.Plot(axes[1],'Predicted')
         else:
             ColorMap_Pred.Plot(axes[1],subtitleIMG)
         # Label 
         from matplotlib.patches import Rectangle
         Classes_Name   =   ColorMap_True.Get_ClasesNamePallete()       # list of string of clases names
         Classes_Color  =   ColorMap_True.Get_ClasesColorPallete()      # list of tuple of (r,g,v) values in (0,1)
         handles = [Rectangle((0,0),1,1, color = tuple((v for v in c))) for c in Classes_Color]
         axes[2].legend(handles,Classes_Name , ncol=1,mode='expand',loc = 'center',frameon=False,fontsize = 'x-large') 
         axes[2].axis('off')
         return plt.show()
     
    @classmethod
    def _imshowMap1(cls,Img,Palette,title = None):
        ColorMap  =   cls(Img,Color_Palette = Palette)
        fig,axes  = plt.subplots(nrows =1,
                             ncols =2,
                             figsize=(50,23),
                             sharex='none',
                             sharey='none')   #img_shape[0]/50,img_shape[1]/50
        fig.subplots_adjust(wspace=0.1,hspace=0.15)
         
        if title is not None:
            fig.suptitle(title,fontsize=20)
        # axes = axes.ravel()  # convert directions of axes to array of dimention 1
        
        ColorMap.Plot(axes[0],None)
        from matplotlib.patches import Rectangle
        Classes_Name   =   ColorMap.Get_ClasesNamePallete()       # list of string of clases names
        Classes_Color  =   ColorMap.Get_ClasesColorPallete()      # list of tuple of (r,g,v) values in (0,1)
        handles = [Rectangle((0,0),1,1, color = tuple((v for v in c))) for c in Classes_Color]
        axes[1].legend(handles,Classes_Name , ncol=1,mode='expand',loc = 'center',frameon=False,fontsize = 'x-large') 
        axes[1].axis('off')
        
        return plt.show()
             
        
        
     
    @staticmethod
    def Check_kwargs(**kwargs):
        My_title = None
        My_subtitleIMG = None
        My_subtitleGt = None
        if 'title' in kwargs:
            My_title = kwargs['title']
        if 'sub_title' in kwargs:
            if len(kwargs['sub_title']) == 2:
                My_subtitleIMG = kwargs['sub_title'][0]
                My_subtitleGt = kwargs['sub_title'][1]
            elif len(kwargs['sub_title']) == 1:
                My_subtitleIMG = kwargs['sub_title'][0]
        return My_title, My_subtitleIMG, My_subtitleGt
    
    @staticmethod
    def Check_Palette(Color_Palette):
        try:
            Pallete = MyCustomColorPalette(Color_Palette)
        except KeyError:
            if Color_Palette is not None:
                print ('No es una Color_Pallete Valida')
            Pallete = None
        return Pallete
    
    @staticmethod
    def Get_MaskImage(Img,Gt):
        Mask = (Gt != -1)*1
        Img = Img + 1;
        Img_Mask = Mask*Img
        Img_Mask = Img_Mask -1
        return Img_Mask

# =============================================================================
# 
# =============================================================================
    @classmethod 
    def imshow_Map(cls,Img,Gt = None, Color_Palette = None, Mask = False,**kwargs):
        '''
        Parameters
        ----------
        Img_True : TYPE = numpy.ndarray
            DESCRIPTION.
            This must be the Ground truth of the Image of size (n_rows,n_cols), it only must have
            N_class diferent values as entries.
        Img_Pred : TYPE: numpy.ndarray
            DESCRIPTION.
            This mus be an Image of Size (n_rows, n_cols), it only must have  at least N_class 
            diferent values as entries.
        Data_Usada : TYPE = sring 
            DESCRIPTION.
            This must be the Name of the image, defined in Database
    
        Returns: A figure With 3 Subplot, 1) Ground Truth, 2) Predicted Image 3) Label of Images
        -------
        None.

        '''

        My_title, My_subtitleIMG, My_subtitleGt = cls.Check_kwargs(**kwargs) 
        Palette = cls.Check_Palette(Color_Palette)
            # =============================================================================
        if (Gt is None):
            cls._imshowMap1(Img,Palette,title = My_title)

        else:
            if Mask:
                Img = cls.Get_MaskImage(Img,Gt)      
            cls._imshowMap2(Img,Gt,Palette, title=My_title, subtitleIMG = My_subtitleIMG,subtitleGt=My_subtitleGt)



# =============================================================================
# MyCustomColorPalete
# =============================================================================
'''
Aqui define se define un diccionario asociado a cada una de las 'famous  hyperspectral images' 
'''
def MyCustomColorPalette(Name_Image):
    MyDict ={ 'IndianP' :  { -1 : ['None',(0,0,0)],
                              0 : ['Alfalfa',(0.86, 0.3712, 0.33999999999999997)],
                              1 : ['Corn-notill',(0.86, 0.5661999999999999, 0.33999999999999997)],
                              2 : ['Corn-mintill',(0.86, 0.7612000000000001, 0.33999999999999997)],
                              3 : ['Corn',(0.7638, 0.86, 0.33999999999999997)],
                              4 : ['Grass-Pasture',(0.5688000000000001, 0.86, 0.33999999999999997)],
                              5 : ['Grass-Trees',(0.3738000000000001, 0.86, 0.33999999999999997)],
                              6 : ['Grass-Pasture-mowed',(0.33999999999999997, 0.86, 0.5012000000000001)],
                              7 : ['Hay-Windrowed',(0.33999999999999997, 0.86, 0.6962000000000002)],
                              8 : ['Oats',(0.33999999999999997, 0.8287999999999999, 0.86)],
                              9 : ['Soybean-notill',(0.33999999999999997, 0.6337999999999998, 0.86)],
                              10: ['Soybean-minitill',(0.33999999999999997, 0.43879999999999986, 0.86)],
                              11: ['Soybean-Clan',(0.43619999999999975, 0.33999999999999997, 0.86)],
                              12: ['Wheat',(0.6311999999999998, 0.33999999999999997, 0.86)],
                              13: ['Woods',(0.8261999999999998, 0.33999999999999997, 0.86)],
                              14: ['Buildings-Grass-Trees-Drives',(0.86, 0.33999999999999997, 0.6987999999999996)],
                              15: ['Stone-Steel-Towers',(0.86, 0.33999999999999997, 0.5037999999999996)] },
             
                'Jasper'  : { 0 : ['Tree',(0, 1, 0)],
                              1 : ['Water',(0, 0, 1)],
                              2 : ['Dirt',(0.435, 0.306, 0.216)],
                              3 : ['Road',(0.5412,0.5843, 0.5922)]},
                
                'Samson'  : { 0 : ['Rock',(0.5412,0.5843, 0.5922)],
                              1 : ['Trees',(0, 1, 0)],
                              2 : ['Water',(0, 0, 1)]},
                            
                'PaviaU'  : {-1 : ['None',(0,0,0)],
                              0 : ['Asphalt',(0.7529411764705882,0.7529411764705882,0.7529411764705882)],
                              1 : ['Meadows',(0.5019607843137255,1,0)],
                              2 : ['Gravel',(0,1,1)],
                              3 : ['Trees',(0,0.6,0)],
                              4 : ['Painted metal sheets',(1,0,1)],
                              5 : ['Bare Soil',(0.6,0,0)],
                              6 : ['Bitumen',(0.6,0.2,1)],
                              7 : ['Self-Blocking Bricks',(1,1,0.4)],
                              8 : ['Shadows',(1,1,0.2)]} 
                } 
    return MyDict[Name_Image]              