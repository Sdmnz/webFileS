#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# module person.py

# module utility.
# ================================================================================================================
#  Modules
# ================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Custom Color Palette 
# =============================================================================
def CustomColorPalette():
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
             
                'Jasper'  : { -1: ['None',(0,0,0)],
                              0 : ['Tree',(0, 1, 0)],
                              1 : ['Water',(0, 0, 1)],
                              2 : ['Dirt',(0.435, 0.306, 0.216)],
                              3 : ['Road',(0.5412,0.5843, 0.5922)]},
                
                'Samson'  : {-1 : ['None',(0,0,0)], 
                              0 : ['Rock',(0.5412,0.5843, 0.5922)],
                              1 : ['Trees',(0, 1, 0)],
                              2 : ['Water',(0, 0, 1)]},
                            
                'PaviaU'  : {-1 : ['None',(0,0,0)],
                              0 : ['Asphalt',(0.7529411764705882,0.7529411764705882,0.7529411764705882)],
                              1 : ['Meadows',(0.5019607843137255,1,0)],
                              2 : ['Gravel',(0,1,1)],
                              3 : ['Trees',(0,0.6,0)],
                              4 : ['Painted metal sheets',(1,0,1)],
                              5 : ['Bare Soil',(0.50196078, 0.25098039, 0.  )],
                              6 : ['Bitumen',(0.6,0.2,1)],
                              7 : ['Self-Blocking Bricks',(0.6,0,0)],
                              8 : ['Shadows',(1,1,0.2)]}, 
                
                'HoustonU': {-1 : ['None',(0,0,0)],
                              0 : ['Healthy grass',(0, 0.803921568627451, 0)],
                              1 : ['Stressed grass',(0.4980392156862745, 0.996108949416342, 0)],
                              2 : ['Synthetic grass',(0.1803921568627451, 0.5450980392156862, 0.3411764705882353)],
                              3 : ['Trees',(0, 0.5450980392156862, 0)],
                              4 : ['Soil',(0.6274509803921569, 0.3215686274509804, 0.17647058823529413)],
                              5 : ['Water',(0.0, 1.0, 1.0)],
                              6 : ['Residential',(0.00392156862745098, 0.00392156862745098, 0.00392156862745098)],
                              7 : ['Commercial',(0.8470588235294118, 0.7490196078431373, 0.8470588235294118)],
                              8 : ['Road',(1.0, 0.0, 0.0)],
                              9 : ['Highway',(0.5450980392156862, 0.0, 0.0)],
                              10: ['Railway',(0.5412,0.5843, 0.5922)],
                              11: ['Parking Lot 1',(1.0, 1.0, 0.0)],
                              12: ['Parking Lot 2',(0.9333333333333333, 0.6039215686274509, 0.0)],
                              13: ['Tennis Court',(0.3333333333333333, 0.10196078431372549, 0.5450980392156862)],
                              14: ['Running Track',(1.0, 0.4980392156862745, 0.3137254901960784)]}   
                    
                } 
    return MyDict     


# ================================================================================================================
#                           Color Map Class
# ================================================================================================================
class Color_Map():
    '''Esta Clase Crea una imagen  RGB (un mapa de color)  de una IMagen en 2D
      Cuyas entradas  enteras que identifican a cada pixel a una clase '''
    Dict_Palete = CustomColorPalette()
    def __init__(self,Img_GT,DataName = None):
        self.Image_Class = Img_GT.astype(int)                            # Se guarda la Imagen para trabajar con ella
        self.rows,self.cols =  Img_GT.shape                           # Dimension Espacial de la Imagen
        self._class_ = np.unique(self.Image_Class)                       # Aqui se define el vector de clases
        self.n_class = self._class_.shape[0]                             # Aqui se calcula el numero de clases
        
        import seaborn as sns  # Este Modulo Me permite llamar una paleta de colores
        mydict = {}
        Palette = np.array(sns.color_palette("hls", self.n_class))
        for cont, i in enumerate(self._class_):
            mydict[i] = ['Class'+str(i),Palette[cont]]
        mydict[-1] = (0,0,0)
        self.color_palette = self.Dict_Palete.get(DataName,mydict)
        self.Mask =  (self.Image_Class != -1)*1
        
    def ImageRGB(self,Img_Pred = None):
        Img_Color =  np.zeros([self.rows,self.cols,3])
        
        if not isinstance(Img_Pred, np.ndarray):
            for i in range(0,self.rows):
                for j in range(0,self.cols):
                    Img_Color[i,j,:] =  self.color_palette.get(self.Image_Class[i,j])[1]
        else:
            for i in range(0,self.rows):
                for j in range(0,self.cols):
                    Img_Color[i,j,:] =  self.color_palette.get(Img_Pred[i,j])[1]
            
        return Img_Color
    
    def Get_ClasesNamePallete(self):
        out = []
        for i,j in self.color_palette.items():
            if i != -1:
                out.append(j[0])
        return out
    
    def Get_ClasesColorPallete(self):
        out = []
        for i,j in self.color_palette.items():
            if i != -1:
                out.append(j[1])
        return out
    
    def plot(self, Img = None, Mask = False, ShowClasses = False,ShowGT = False, **kwargs):
        Title =  kwargs.get('Title')
        Subtitle_GT = kwargs.get('Subtitle_GT')
        Subtitle_Img = kwargs.get('Subtitle_Img')
        
        GT_RGB =  self.ImageRGB()
        if isinstance(Img, np.ndarray):
            if Mask: 
                Img = (Img+1) * self.Mask
                Img = Img -1
            Img_RGB = self.ImageRGB(Img_Pred = Img)
            
        if ShowClasses == True:
            from matplotlib.patches import Rectangle
            Classes_Name   =   self.Get_ClasesNamePallete()       # list of string of clases names
            Classes_Color  =   self.Get_ClasesColorPallete()      # list of tuple of (r,g,v) values in (0,1)
            handles = [Rectangle((0,0),1,1, color = tuple((v for v in c))) for c in Classes_Color]
            
        
        # =============================================================================
        #       Conteo De casos      
        # =============================================================================
        if not isinstance(Img, np.ndarray)  and ShowClasses == False:
            #  Case 1
            fig,axes  = plt.subplots(nrows =1,
                                     ncols=1,
                                     figsize=(50,23),
                                     sharex='none',
                                     sharey='none')
            fig.suptitle(Title,fontsize=20)
            axes.imshow(GT_RGB)
            axes.axis('off')
            axes.set_title(Subtitle_GT,fontsize=25)
            plt.show()
            
        elif isinstance(Img, np.ndarray) and ShowClasses == False:
            # Case 2
            if not ShowGT:
                fig,axes  = plt.subplots(nrows =1,
                                         ncols=1,
                                         figsize=(50,23),
                                         sharex='none',
                                         sharey='none')
                fig.suptitle(Title,fontsize=20)
                axes.imshow(Img_RGB)
                axes.axis('off')
                axes.set_title(Subtitle_Img,fontsize=25)
                plt.show()
            else:
            # Case 3
                fig,axes  = plt.subplots(nrows =1,
                                     ncols=2,
                                     figsize=(50,23),
                                     sharex='none',
                                     sharey='none')
                fig.suptitle(Title,fontsize=20)
                fig.subplots_adjust(wspace=0.1,hspace=0.15)
                axes.ravel()
                axes[0].imshow(GT_RGB)
                axes[0].axis('off')
                axes[0].set_title(Subtitle_GT,fontsize=25)
                axes[1].imshow(Img_RGB)
                axes[1].axis('off')
                axes[1].set_title(Subtitle_Img ,fontsize=25)
                plt.show()
                
        elif not isinstance(Img, np.ndarray) and ShowClasses == True:
            # Case 4
            fig,axes  = plt.subplots(nrows =1,
                                     ncols=2,
                                     figsize=(50,23),
                                     sharex='none',
                                     sharey='none')
            fig.suptitle(Title,fontsize=20)
            fig.subplots_adjust(wspace=0.1,hspace=0.15)
            axes.ravel()
            axes[0].imshow(GT_RGB)
            axes[0].axis('off')
            axes[0].set_title(Subtitle_GT,fontsize=25)
            axes[1].legend(handles,
                           Classes_Name,
                           ncol=1,
                           mode='expand',
                           loc = 'center',
                           frameon=False,
                           fontsize = 'x-large')
            axes[1].axis('off')
            plt.show()
        elif isinstance(Img, np.ndarray) and ShowClasses == True:
            # Case 5
            if not ShowGT:
                fig,axes  = plt.subplots(nrows =1,
                                         ncols=2,
                                         figsize=(50,23),
                                         sharex='none',
                                         sharey='none')
                fig.suptitle(Title,fontsize=20)
                fig.subplots_adjust(wspace=0.1,hspace=0.15)
                axes.ravel()
                axes[0].imshow(Img_RGB)
                axes[0].axis('off')
                axes[0].set_title(Subtitle_Img,fontsize=25)
                axes[1].legend(handles,
                               Classes_Name,
                               ncol=1,
                               mode='expand',
                               loc = 'center',
                               frameon=False,
                               fontsize = 'x-large')
                axes[1].axis('off')
                plt.show()

            else:
            # Case 6
                fig,axes  = plt.subplots(nrows =1,
                                     ncols=3,
                                     figsize=(50,23),
                                     sharex='none',
                                     sharey='none')
                fig.suptitle(Title,fontsize=20)
                fig.subplots_adjust(wspace=0.1,hspace=0.15)
                axes.ravel()
                axes[0].imshow(GT_RGB)
                axes[0].axis('off')
                axes[0].set_title(Subtitle_Img,fontsize=25)
                axes[1].imshow(Img_RGB)
                axes[1].axis('off')
                axes[1].set_title(Subtitle_GT,fontsize=25)
                axes[2].legend(handles,
                               Classes_Name,
                               ncol=1,
                               mode='expand',
                               loc = 'center',
                               frameon=False,
                               fontsize = 'x-large')
                axes[2].axis('off')
                plt.show() 
        
            

             
            
        
        
        
        
            
        # elif isinstance(Img, np.ndarray) and showClasses == False:
            
            
        # if Title is not None:
        #     axes.set_title(Title,fontsize=25)
        

        



        
        
    
    
    
    
    

# def imshow_Map(Img,Gt = None, DataName = None, Mask = False, showClasses = False, **kwargs):
#     if GT == None and DataName == None and Mask == None and showClasses == False:
#         _imshowMap1(Img,**kwargs)
        
        
#     if (Gt is None):
#         cls._imshowMap1(Img,Palette,**kwargs)

#     else:
#         if Mask:
#             Img = cls.Get_MaskImage(Img,Gt)      
#         cls._imshowMap2(Img,Gt,Palette)
#         plt.show()
    
    
    

#     def Plot(self,axes,Title = 'None'):
#         # Asegurarse importar import matplotlib.pyplot as plt
#         axes.imshow(self.Img_Color)
#         axes.axis('off')
#         if Title is not None:
#             axes.set_title(Title,fontsize=25)
   
#     @classmethod
#     def _imshowMap2(cls,Img,Gt,Color_Palette,title=None,subtitleIMG =None,subtitleGt = None):
#          # import matplotlib.pyplot as plt
#          ColorMap_True  =   cls(Gt,Color_Palette = Color_Palette)      # Creando el Objeto Color Map
#          ColorMap_Pred  =   cls(Img,Color_Palette = Color_Palette)       # Creando el Objeto Color Map
#          fig,axes  = plt.subplots(nrows =1,
#                              ncols =3,
#                              figsize=(50,23),
#                              sharex='none',
#                              sharey='none')   #img_shape[0]/50,img_shape[1]/50
#          fig.subplots_adjust(wspace=0.1,hspace=0.15)
         
#          if title ==None:
#              fig.suptitle('Original vs Predicted',fontsize=20)
#          else:
#              fig.suptitle(title,fontsize=20)
             
#          axes = axes.ravel()  # convert directions of axes to array of dimention 1
#          # Ground Truth
#          if subtitleGt is None:
#              ColorMap_True.Plot(axes[0],'Ground Truth')
#          else:
#              ColorMap_True.Plot(axes[0],subtitleGt)
#          # Predicted Image
#          if subtitleIMG is None:
#              ColorMap_Pred.Plot(axes[1],'Predicted')
#          else:
#              ColorMap_Pred.Plot(axes[1],subtitleIMG)
#          # Label 
#          from matplotlib.patches import Rectangle
#          Classes_Name   =   ColorMap_True.Get_ClasesNamePallete()       # list of string of clases names
#          Classes_Color  =   ColorMap_True.Get_ClasesColorPallete()      # list of tuple of (r,g,v) values in (0,1)
#          handles = [Rectangle((0,0),1,1, color = tuple((v for v in c))) for c in Classes_Color]
#          axes[2].legend(handles,Classes_Name , ncol=1,mode='expand',loc = 'center',frameon=False,fontsize = 'x-large') 
#          axes[2].axis('off')
#          return plt.show()
     
#     @classmethod
#     def _imshowMap1(cls,Img,Palette,title = None):
#         ColorMap  =   cls(Img,Color_Palette = Palette)
#         fig,axes  = plt.subplots(nrows =1,
#                              ncols =2,
#                              figsize=(50,23),
#                              sharex='none',
#                              sharey='none')   #img_shape[0]/50,img_shape[1]/50
#         fig.subplots_adjust(wspace=0.1,hspace=0.15)
         
#         if title is not None:
#             fig.suptitle(title,fontsize=20)
#         # axes = axes.ravel()  # convert directions of axes to array of dimention 1
        
#         ColorMap.Plot(axes[0],None)
#         from matplotlib.patches import Rectangle
#         Classes_Name   =   ColorMap.Get_ClasesNamePallete()       # list of string of clases names
#         Classes_Color  =   ColorMap.Get_ClasesColorPallete()      # list of tuple of (r,g,v) values in (0,1)
#         handles = [Rectangle((0,0),1,1, color = tuple((v for v in c))) for c in Classes_Color]
#         axes[1].legend(handles,Classes_Name , ncol=1,mode='expand',loc = 'center',frameon=False,fontsize = 'x-large') 
#         axes[1].axis('off')
        
#         return plt.show()
             
        
        
    
#     @staticmethod
#     def Check_Palette(Color_Palette):
#         try:
#             Pallete = MyCustomColorPalette(Color_Palette)
#         except KeyError:
#             if Color_Palette is not None:
#                 print ('No es una Color_Pallete Valida')
#             Pallete = None
#         return Pallete
    
#     @staticmethod
#     def Get_MaskImage(Img,Gt):
#         Mask = (Gt != -1)*1
#         Img = Img + 1;
#         Img_Mask = Mask*Img
#         Img_Mask = Img_Mask -1
#         return Img_Mask


    
    
    
    
    
    
    
    
    
    
    
    



     