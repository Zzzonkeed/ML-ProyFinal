################# OBTENCION DE LOS DATOS ################
install.packages("gsheet")
library(gsheet)
library(readr)
URL<-"https://docs.google.com/spreadsheets/d/1cJZIUTPoYliSej8ZQumPdrvWjcZeRyuEqyC4JyXpx-8/edit?usp=sharing"
data<-read_csv(construct_download_url(URL))

#dimension de los datos RAW [1]  42 481
dim(data)

#se toman 42 numeros al azar
s<-sample(1:42,size = 42,replace=FALSE)

#Datos tomados al azar
#[1] 15 16 42 30 10 13 32 40 26 35  8 25 24 23 27 29  5 22  6 36 12 21  7 19 39  9 38  1 11  2 41
#[32] 20 34  4  3 37 33 31 28 14 18 17
s

#se desordenan los datos utilizando los numeros al azar de S
desor<-data[s,]
desor

#se toma la variable dependiente Y()
Y<-desor[,1]
Y

#se crea la matriz con las variables independientes 
X<-desor[,-c(1)]
X

############ DETECCION DE DATOS FALTANTES ############
library("mlbench")
library(naniar)
library(visdat)

vis_dat(y_norm)
vis_dat(x_norm)

#Datos que dan NAN al normalizar, datos iguales entre si
#son X122, X127, X202, X205 al X210
boxplot(X$X122)
X$X122 #0.24
X$X127 #0.25
X$X202 #0.06
X$X205 #0.07
X$X206 #0.06
X$X207 #0.07
X$X208 #0.06
X$X209 #0.06
X$X210 #0.07

#eliminacion columnas de datos iguales
x_new<-X[,-c(122,127,202,205:210)]


############ ELIMINACION DE LAS CORRELACIONES ############
install.packages("caret")
library(caret)

corre_matrix<-cor(x_new)
index <- findCorrelation(corre_matrix, .80)
x_clean<-x_new[,-c(index)]


############ NORMALIZACION DE LOS DATOS ############

str(X)
#https://rpubs.com/sediaz/Normalize source
#normalizar los datos de las columnas
x_norm <- as.data.frame(apply(x_clean, 2, function(z) (z - min(z))/(max(z)-min(z))))
y_norm<- as.data.frame(apply(Y, 2, function(z) (z - min(z))/(max(z)-min(z))))


############ DETECCION DE DATOS ATIPICOS ############
dim(x_norm)
#[1]  42 107
boxplot(x_norm)

#-----------------------------------------------------------------
#--- Se detectan los outliers y se remplazan por la mediana
install.packages("outliers")
library(outliers)
rm.outlier(x_norm, fill=TRUE, median=TRUE, opposite = FALSE)
#-----------------------------------------------------------------

#----------- Se vuelve a normalizar los datos -----------
#----------- luego de extraer los outliers --------------
#--------------------------------------------------------

xDataNorma <- as.data.frame(apply(x_norm, 2, function(z) (z - min(z))/(max(z)-min(z))))
yDataNorma <- as.data.frame(apply(y_norm, 2, function(z) (z - min(z))/(max(z)-min(z))))


############ VISUALIZACION DE LOS DATOS ############
#-----------------------------------------------------------------
#--- PCA
install.packages(c("FactoMineR", "factoextra"))
library("FactoMineR")
library("factoextra")
head(xDataNorma[,1:7])

x.pca<-PCA(xDataNorma, scale.unit=TRUE, ncp=42, graph=FALSE)
print(x.pca)
summary(x.pca)
names(x.pca)

#-----------------------------------------------------------------
#biblot
fviz_pca_biplot(x.pca, repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)
#-----------------------------------------------------------------

#------------- eigenvalue ----------------------------------------
x.eig<-get_eigenvalue(x.pca)
x.eig

#plot del procentaje de relevancia de las intancias
#el porcentaje corresponde a las varianzas
fviz_eig(x.pca, 
         addlabels = TRUE,
         title = "Porcentaje de relevancia para cada instancia",
         xlab = "Principal Component",
         ylab = "% of variances",
         ncp=15, 
         ylim = c(0, 19), 
         linecolor = "red")
#-----------------------------------------------------------------

#------------- Graficos de las Variables -------------------------
variables<-get_pca_var(x.pca)
variables

# Coordinates
head(variables$coord)
# Cos2: quality on the factore map
head(variables$cos2)
# Contributions to the principal components
head(variables$contrib,5)

# Contributions of variables to PC1
fviz_contrib(x.pca, choice = "var", axes = 1)
# Contributions of variables to PC2
fviz_contrib(x.pca, choice = "var", axes = 2)
# Total contribution to PC1 and PC2
# grafico para ver la contribucion total de los
# principales atributos hacia las variables 
# DIM1 y DIM2
fviz_contrib(x.pca, choice = "var", axes = 1:2)

fviz_pca_var(x.pca, col.var = "contrib",
             title = "Contribución de las variables Dim1 y Dim2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             select.var = list(name = NULL, cos2 = NULL, contrib = NULL)
)
#-----------------------------------------------------------------

#------------ Descripcion de la Dimension ------------------------
x.desc <- dimdesc(x.pca, axes = c(1,2), proba = 0.05)
# Description of dimension 1
x.desc$Dim.1

# Description of dimension 2
x.desc$Dim.2
#-----------------------------------------------------------------

############ REDUCCION DE LA DIMENSION USANDO PCA ############
# se dejan las instancias que suman el 98.21660%, 
# que corresponden a los datos con mayor relevancia
x_reduc<-xDataNorma[-c(6,16,19,20,23,27,28,29,33,41),]
x_reduc

y_reduc<-yDataNorma[-c(6,16,19,20,23,27,28,29,33,41),]
y_reduc

boxplot(x_reduc)
############ REGRESION ############

#----------------- Regresion Lineal -----------------
install.packages("tidyverse")
library(tidyverse)
library(ggpubr)
theme_set(theme_pubr())

#se particiona los datos en 2 dataset
#el primero se utiliza para entrenar y el segundo para validar
set.seed(100)
train_data_index<-sample(1:nrow(x_reduc),0.8*nrow(x_reduc))
train_data<-x_reduc[train_data_index,]
test_data<-x_reduc[-train_data_index,]

#----------------- lm metodo -----------------
#modelo lineal con solo un atributo princial
lmodel<-lm(y_reduc~x_reduc$X19, data = train_data)

#---------------------------------------------------------------
#modelo lineal con los principales atributos
#vease el grafico de:
# Contribucion total entre DIM1 y DIM2 de los atributos
#fviz_contrib(x.pca, choice = "var", axes = 1:2)

#lmodel<-lm(y_reduc~x_reduc$X19+x_reduc$X13+x_reduc$X400+
#             x_reduc$X152+x_reduc$X480+x_reduc$X164,
#                 data=train_data)
#---------------------------------------------------------------

lmodel
#datos del modelo lineal
summary(lmodel)
summary(lmodel)$coefficient

fitted(lmodel) # predicted values

#prediccion del modelo
distPred <- predict(lmodel, test_data)
distPred

#grafico de la primera variable
plot(distPred,y_reduc,
     main="Regresion Lineal",
     xlab = "y predecida",
     ylab = "Y(x)",
     col = "black",
     las=1,
     pch = 19)   

#dibujar el modelo lineal en el grafico
abline(lmodel, lwd=3, col="red")
#---------------------------------------------------------
#-------- caret metodo -----------------------------------
# en proceso...

lmodel_c<-train(y_reduc~.,data=train_data, method="lm")

#---------------------------------------------------------
#--- Regresion Polinomial ---
#modelo polinomial de segundo grado con un atributo
#pmodel_2<-lm(y_reduc~poly(x_reduc$X19,degree=2,raw=T))

#modelo polinomial con los 6 primeros atributos mas relevantes
pmodel_2<-lm(y_reduc ~ polym(x_reduc$X19, x_reduc$X13, 
                             x_reduc$X400, x_reduc$X152,
                             x_reduc$X480, x_reduc$X164,
                             degree=2, raw=TRUE))

pmodel_2
summary(pmodel_2)

#prediccion del modelo
distPred_poly <- predict(pmodel_2, test_data)
distPred_poly

#---------------------------------------------------------
#grafico del modelo polinomial
plot(distPred_poly,y_reduc,
     main="Regresion Polinomial ^2 Multiple variable",
     xlab = "Y predicha",
     ylab = "Y(x)",
     col = "black",
     las=1,
     pch = 19)  

#se imprime la curva del modelo polinomial
lines(smooth.spline(distPred_poly, predict(pmodel_2)), col="blue", lwd=3)

#se comparan ambos modelos, el lineal y el polinomial de segundo grado
anova(lmodel, pmodel_2)

#-----------------------------------------------------------------
#se crea un tercer modelo polonimial de tercer grado
pmodel_3<-lm(y_reduc~poly(x_reduc$X19,degree = 3,raw = T))

summary(pmodel_3)

#se imprime en el grafico del modelo de tercer grado
lines(smooth.spline(x_reduc$X13, predict(pmodel_3)), col="orange", lwd=3)
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#se crea un cuarto modelo polinomial de cuarto grado
pmodel_4<-lm(y_reduc~poly(x_reduc$X19,degree = 4,raw=F))

summary(pmodel_4)
#se imprime la curva del modelo de cuarto grado
lines(smooth.spline(x_reduc$X13, predict(pmodel_4)), col="green", lwd=3)


#################### Medidas de desempeño ######################
#-------- R squared --------

#-------- usando caret -----
#---------------------------------------------------------
#### modelo lineal
actual_lm<-y_reduc
predicted_lm<-lmodel$fitted.values

summary(lmodel)
#--- R-squared modelo lineal
R2(predicted_lm, actual_lm)
#--- RMSE modelo lineal
RMSE(predicted_lm, actual_lm)

#---------------------------------------------------------
#### modelo polinomial de segundo grado
actual_pl2<-y_reduc
predicted_pl2<-pmodel_2$fitted.values

summary(pmodel_2)
#--- R-squared modelo polinomial ^2
R2(predicted_pl2, actual_pl2)
#--- RMSE modelo polinomial ^2
RMSE(predicted_pl2, actual_pl2)

#---------------------------------------------------------
#### modelo polinomial de tercer grado
actual_pl3<-y_reduc
predicted_pl3<-pmodel_3$fitted.values

summary(pmodel_3)
#--- R-squared modelo polinomial ^3
R2(predicted_pl3, actual_pl3)
#--- RMSE modelo polinomial ^3
RMSE(predicted_pl3, actual_pl3)
#---------------------------------------------------------

############# Validacion Cruzada #########################
#------------ K-fold coss validation ---------------------
set.seed(101)
#numero de observaciones
N<-nrow(x_reduc)

#numero de particiones K
folds <- 4 #2 - 4 - 8 - 16

#se generan los indices de las particiones
kfold<-split(sample(1:N),1:folds)
str(kfold)
#List of 4
#$ 1: int [1:8] 15 13 1 28 23 6 20 30
#$ 2: int [1:8] 26 21 11 7 19 16 29 32
#$ 3: int [1:8] 25 17 5 3 12 9 4 14
#$ 4: int [1:8] 24 18 8 22 2 27 10 31

#se verifica que no se repitan las instancias
#qeu aparezcan exactamente 1 vez en el dataset kfold
kfold %>% unlist() %>% length() == N
#[1] TRUE -------> todo OK

######### K1 ##################
#modelo polinomial con los 2 primeros atributos mas relevantes
#se utiliza los fold 2, 3 y 4 para el entrenamiento
polymodel_k1<-lm(y_reduc ~ polym(x_reduc$X19, x_reduc$X13,
                             degree=2, raw=TRUE), 
            data = train_data[-kfold$`1`,])
summary(polymodel_k1)

#prediccion del modelo
polyPred_k1 <- predict(polymodel_k1, 
                            test_data[-c(kfold$`2`,
                                         kfold$`3`,
                                         kfold$`4`),])
polyPred_k1

#se imprime la curva del modelo polinomial
lines(smooth.spline(polyPred_k1, predict(polymodel_k1)), col="blue", lwd=3)


######### K2 ##################
#modelo polinomial con los 2 primeros atributos mas relevantes
#se utiliza los fold 2, 3 y 4 para el entrenamiento
polymodel_k2<-lm(y_reduc ~ polym(x_reduc$X19, x_reduc$X13,
                                 degree=2, raw=TRUE), 
                 data = train_data[-kfold$`2`,])
summary(polymodel_k2)

#prediccion del modelo
polyPred_k2 <- predict(polymodel_k2, 
                       test_data[-c(kfold$`1`,
                                    kfold$`3`,
                                    kfold$`4`),])
polyPred_k2

#se imprime la curva del modelo polinomial
lines(smooth.spline(polyPred_k2, predict(polymodel_k2)), col="green", lwd=3)
